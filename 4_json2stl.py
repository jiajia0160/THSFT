import torch
import json
import numpy as np
from stlcgpp.formula import *
from stlcgpp.tests import *

'''
1、读取train.json文件
2、对PWL轨迹进行线性插值
3、将JSON结构化描述转换为stlcg++的STL表达式
4、计算鲁棒性分数
'''

# 定义房间边界
ROOMS = {
    'RestRoom': (0, 3, 7.5, 10.0),
    'MasterBedroom': (7, 10, 5.5, 10.0),
    'RestRoom2': (7, 10, 2, 4),
    'ExerciseRoom': (4, 6, 8, 10),
    'LivingRoom': (2, 5, 3, 6),
    'Kitchen': (0, 1, 0, 2),
    'DiningRoom': (2, 4, 0, 1),
    'Bedroom': (5, 10, 0, 2)
}

class STLConverter:
    def __init__(self, rooms_dict):
        self.rooms = rooms_dict
        
    def interpolate_trajectory(self, pwl, dt=0.1):
        """
        将PWL轨迹线性插值为均匀时间步
        
        Args:
            pwl: [[x1,y1,t1], [x2,y2,t2], ...] 格式的轨迹
            dt: 时间步长
            
        Returns:
            torch.Tensor: shape [T, 2] 的均匀时间步轨迹
        """
        pwl = np.array(pwl)
        t_start = pwl[0, 2]
        t_end = pwl[-1, 2]
        
        # 生成均匀时间步
        t_uniform = np.arange(t_start, t_end + dt, dt)
        
        # 对x和y分别进行线性插值
        x_interp = np.interp(t_uniform, pwl[:, 2], pwl[:, 0])
        y_interp = np.interp(t_uniform, pwl[:, 2], pwl[:, 1])
        
        # 转换为torch tensor
        trajectory = torch.tensor(np.stack([x_interp, y_interp], axis=1), 
                                 dtype=torch.float32)
        
        return trajectory, t_uniform
    
    def get_room_name(self, item):
        """尝试从字典中获取房间名"""
        for key in ['target', 'room', 'area', 'location', 'nearest_room']:
            if key in item and item[key] is not None:
                return item[key]
        return None

    def create_room_predicate(self, room_name):
        """
        创建房间谓词函数，检查位置是否在房间内
        
        Args:
            room_name: 房间名称
            
        Returns:
            函数：输入[T, 2]的轨迹，输出[T]的布尔值
        """
        if room_name not in self.rooms:
            # 如果房间名不在预定义字典中，尝试查找是否有匹配的键（忽略大小写或部分匹配）
            # 这里简单处理：如果找不到，抛出更具体的错误
            raise ValueError(f"Unknown room: {room_name}")

        x_min, x_max, y_min, y_max = self.rooms[room_name]
        
        def in_room(states):
            """
            states: [T, 2] tensor
            返回: [T] tensor，正值表示在房间内，负值表示在房间外
            """
            x, y = states[:, 0], states[:, 1]
            
            # 计算到房间边界的最小距离（作为鲁棒性度量）
            # 在房间内时为正，在房间外时为负
            dx_min = x - x_min
            dx_max = x_max - x
            dy_min = y - y_min
            dy_max = y_max - y
            
            # 使用最小距离作为鲁棒性
            robustness = torch.min(torch.min(dx_min, dx_max), 
                                  torch.min(dy_min, dy_max))
            return robustness
        
        return Predicate(f"in_{room_name}", predicate_function=in_room)
    
    def time_to_steps(self, time_bound, dt=0.1):
        """
        将时间转换为时间步数
        """
        if time_bound is None:
            return None
        if isinstance(time_bound, list):
            return [int(t / dt) for t in time_bound]
        return int(time_bound / dt)
    
    def build_stl_formula(self, stl_json, dt=0.1):
        """
        将JSON结构转换为STL公式
        
        Args:
            stl_json: 结构化的JSON描述
            dt: 时间步长
            
        Returns:
            STL formula对象
        """
        formulas = []
        
        # 处理temporal_constraints
        if 'temporal_constraints' in stl_json:
            for constraint in stl_json['temporal_constraints']:
                try:
                    formula = self._build_constraint(constraint, dt)
                    if formula is not None:
                        formulas.append(formula)
                except Exception as e:
                    print(f"Warning: Failed to build constraint: {e}")
        
        # 处理global_constraints
        if 'global_constraints' in stl_json:
            for constraint in stl_json['global_constraints']:
                try:
                    formula = self._build_global_constraint(constraint, dt)
                    if formula is not None:
                        formulas.append(formula)
                except Exception as e:
                    print(f"Warning: Failed to build global constraint: {e}")
        
        # 合并所有公式
        if len(formulas) == 0:
            raise ValueError("No valid constraints found")
        elif len(formulas) == 1:
            return formulas[0]
        else:
            # 使用 & 连接所有约束
            combined = formulas[0]
            for f in formulas[1:]:
                combined = combined & f
            return combined
    
    def _build_constraint(self, constraint, dt):
        """构建单个约束"""
        if constraint['type'] == 'sequence':
            return self._build_sequence(constraint, dt)
        elif constraint['type'] == 'choice':
            return self._build_choice(constraint, dt)
        return None
    
    def _build_sequence(self, constraint, dt):
        """
        构建顺序约束
        sequence: 
        1. 满足所有绝对时间约束 (F[0, t1] A) & (F[0, t2] B) ...
        2. 满足顺序约束 F(A & F(B & ...))
        """
        tasks = constraint['tasks']
        
        # 1. 构建绝对时间约束
        absolute_constraints = []
        for task in tasks:
            room_name = self.get_room_name(task)
            if not room_name:
                continue
                
            room_pred = self.create_room_predicate(room_name)
            
            # 构建基本谓词（包含wait）
            # 如果有wait，谓词变为: in_room & G[0, wait] in_room
            base_pred = room_pred > 0
            if 'actions' in task:
                for action in task['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time is not None and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            # 必须在房间内，并且从那一刻起保持在房间内
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])

            # 添加时间约束
            if 'time_bound' in task and task['time_bound'] is not None:
                time_bound = task['time_bound']
                if isinstance(time_bound, list):
                    interval = self.time_to_steps(time_bound, dt)
                    # 稍微放宽一点时间界限，防止边界误差
                    if isinstance(interval, list) and len(interval) == 2:
                        interval[1] += 1
                    # F[t_start, t_end] (base_pred)
                    absolute_constraints.append(Eventually(base_pred, interval=interval))
                else:
                    steps = self.time_to_steps(time_bound, dt)
                    # F[0, t_end] (base_pred)
                    # 稍微放宽一点时间界限
                    absolute_constraints.append(Eventually(base_pred, interval=[0, steps + 1]))
            else:
                # 如果没有时间约束，也需要最终到达
                absolute_constraints.append(Eventually(base_pred))

        # 2. 构建顺序约束 (从后往前)
        # F(P1 & F(P2 & F(P3)))
        ordering_formula = None
        for i in range(len(tasks) - 1, -1, -1):
            task = tasks[i]
            room_name = self.get_room_name(task)
            if not room_name:
                continue
            
            room_pred = self.create_room_predicate(room_name)
            
            # 基本谓词 (同上)
            base_pred = room_pred > 0
            if 'actions' in task:
                for action in task['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time is not None and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])
            
            if ordering_formula is None:
                # 最后一个任务
                ordering_formula = Eventually(base_pred)
            else:
                # 前面的任务：F(current & ordering_formula)
                ordering_formula = Eventually(base_pred & ordering_formula)
        
        # 3. 组合所有约束
        final_formula = ordering_formula
        for c in absolute_constraints:
            final_formula = final_formula & c
            
        return final_formula
    
    def _build_choice(self, constraint, dt):
        """
        构建选择约束
        choice: φ1 ∨ φ2 ∨ ... (使用 | 操作符)
        """
        options = constraint['options']
        formulas = []
        
        for option in options:
            room_name = self.get_room_name(option)
            if not room_name:
                continue
                
            room_pred = self.create_room_predicate(room_name)
            
            # 基本谓词
            base_pred = room_pred > 0
            if 'actions' in option:
                for action in option['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time is not None and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])

            # 创建到达房间的公式
            if 'time_bound' in option and option['time_bound'] is not None:
                time_bound = option['time_bound']
                if isinstance(time_bound, list):
                    interval = self.time_to_steps(time_bound, dt)
                    reach_room = Eventually(base_pred, interval=interval)
                else:
                    steps = self.time_to_steps(time_bound, dt)
                    reach_room = Eventually(base_pred, interval=[0, steps])
            else:
                reach_room = Eventually(base_pred)
            
            formulas.append(reach_room)
        
        # 使用 | 操作符连接所有选项
        if len(formulas) == 0:
             raise ValueError("No valid options in choice constraint")
        elif len(formulas) == 1:
            return formulas[0]
        else:
            combined = formulas[0]
            for f in formulas[1:]:
                combined = combined | f
            return combined
    
    def _build_global_constraint(self, constraint, dt):
        """构建全局约束"""
        if 'condition' not in constraint:
             raise ValueError("Global constraint missing 'condition'")
             
        room_name = self.get_room_name(constraint['condition'])
        if not room_name:
             raise ValueError(f"No room specified in global constraint: {constraint}")
             
        room_pred = self.create_room_predicate(room_name)
        
        if constraint['type'] == 'always_avoid':
            # 始终避免进入某房间：G(¬in_room)
            return Always(room_pred < 0)
        elif constraint['type'] == 'always_satisfy':
            # 始终满足在某房间：G(in_room)
            return Always(room_pred > 0)
        
        return None
    
    def compute_robustness(self, stl_json, pwl, dt=0.1, 
                          approx_method="true", temperature=1.0):
        """
        计算轨迹对STL公式的鲁棒性
        
        Args:
            stl_json: 结构化JSON描述
            pwl: PWL轨迹
            dt: 时间步长
            approx_method: "true", "logsumexp", 或 "softmax"
            temperature: 平滑参数
            
        Returns:
            robustness_value: 标量鲁棒性值
            robustness_trace: 鲁棒性轨迹
            trajectory: 插值后的轨迹
        """
        # 插值轨迹
        trajectory, t_uniform = self.interpolate_trajectory(pwl, dt)
        
        # 构建STL公式
        formula = self.build_stl_formula(stl_json, dt)
        
        # 计算鲁棒性
        robustness_trace = formula(trajectory)
        
        # 使用t=0时刻的鲁棒性作为整个轨迹的鲁棒性分数
        # formula.robustness() 默认返回整个trace的最小值，这对于Eventually等公式是不合适的
        robustness_value = robustness_trace[0]
        
        return {
            'robustness_value': robustness_value.item(),
            'robustness_trace': robustness_trace.detach().numpy(),
            'trajectory': trajectory.detach().numpy(),
            'time': t_uniform
        }
    
    def compute_gradient(self, stl_json, pwl, dt=0.1,
                        approx_method="logsumexp", temperature=1.0):
        """
        计算鲁棒性对轨迹的梯度（用于优化）
        
        Args:
            stl_json: 结构化JSON描述
            pwl: PWL轨迹
            dt: 时间步长
            approx_method: "logsumexp" 或 "softmax"
            temperature: 平滑参数
            
        Returns:
            gradient: 梯度数组
            robustness_value: 鲁棒性值
        """
        # 插值轨迹
        trajectory, t_uniform = self.interpolate_trajectory(pwl, dt)
        trajectory.requires_grad = True
        
        # 构建STL公式
        formula = self.build_stl_formula(stl_json, dt)
        
        # 计算梯度
        gradient = torch.func.grad(formula.robustness)(
            trajectory,
            approx_method=approx_method,
            temperature=temperature
        )
        
        robustness_value = formula.robustness(
            trajectory,
            approx_method=approx_method,
            temperature=temperature
        )
        
        return {
            'gradient': gradient.detach().numpy(),
            'robustness_value': robustness_value.item(),
            'trajectory': trajectory.detach().numpy(),
            'time': t_uniform
        }


def process_train_data(json_file, output_file, dt=0.1):
    """
    处理训练数据文件
    
    Args:
        json_file: train.json文件路径
        output_file: 输出文件路径
        dt: 时间步长
        
    Returns:
        results: 包含每个样本鲁棒性的列表
    """
    # 读取数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建转换器
    converter = STLConverter(ROOMS)
    
    results = []
    
    for idx, sample in enumerate(data):
        try:
            stl_json_str = sample['stl_json']
            # 解析STL JSON字符串
            if isinstance(stl_json_str, str):
                if "```json" in stl_json_str:
                    stl_json_str = stl_json_str.split("```json")[1].split("```")[0]
                elif "```" in stl_json_str:
                    stl_json_str = stl_json_str.split("```")[1].split("```")[0]
                stl_json = json.loads(stl_json_str.strip())
            else:
                stl_json = stl_json_str
                
            pwl = sample['pwl']
            
            print(f"\nProcessing sample {idx + 1}/{len(data)}...")
            
            # 计算鲁棒性
            result = converter.compute_robustness(stl_json, pwl, dt)
            
            # 记录鲁棒性分数到样本中
            sample['robustness_score'] = result['robustness_value']
            
            result['sample_id'] = idx
            result['stl_json'] = stl_json
            results.append(result)
            
            print(f"Robustness value: {result['robustness_value']:.4f}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # 失败时记录None
            sample['robustness_score'] = None
            continue
            
    # 保存带有鲁棒性分数的新数据文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved processed data to {output_file}")
    
    return results


# 使用示例
if __name__ == "__main__":
    input_file = '/home/lijia/code/1113_CLHS/train_dataset/train_low_with_stl_output_with_target_and_pwl.json'
    output_file = '/home/lijia/code/1113_CLHS/train_dataset/train_low_with_stl_output_with_target_and_pwl_robustness.json'
    
    # 处理训练数据
    results = process_train_data(input_file, output_file, dt=0.1)
    
    # 保存详细结果（可选）
    with open('robustness_results.json', 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        for r in results:
            r['robustness_trace'] = r['robustness_trace'].tolist()
            r['trajectory'] = r['trajectory'].tolist()
            r['time'] = r['time'].tolist()
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed {len(results)} samples successfully!")
    
    # 示例：计算单个样本的梯度
    if len(results) > 0:
        converter = STLConverter(ROOMS)
        sample = results[0]
        
        gradient_result = converter.compute_gradient(
            sample['stl_json'],
            [[1, 3, 0], [3, 0.5, 5], [0.5, 1, 10], [7.5, 1, 15]],
            dt=0.1,
            approx_method="logsumexp",
            temperature=1.0
        )
        
        print("\nGradient shape:", gradient_result['gradient'].shape)
        print("Gradient (first 5 steps):\n", gradient_result['gradient'][:5])