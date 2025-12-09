import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
import requests
import os

'''
想让claude补充choice的处理来着，1204

它的判断逻辑就是：
根据json里面
'''

def parse_stl_json(stl_json_str: str) -> Dict:
    """解析STL JSON字符串"""
    if "```json" in stl_json_str:
        stl_json_str = stl_json_str.split("```json")[1].split("```")[0]
    return json.loads(stl_json_str.strip())

def calculate_distance(p1: List[float], p2: List[float]) -> float:
    """计算两点之间的欧几里得距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_room_center(room_info: Dict) -> List[float]:
    """获取房间中心坐标"""
    x_start, x_end, y_start, y_end = room_info['position and size']
    return [(x_start + x_end) / 2, (y_start + y_end) / 2]

def is_point_in_room(point: List[float], room_info: Dict) -> bool:
    """判断点是否在房间内"""
    x, y = point
    x_start, x_end, y_start, y_end = room_info['position and size']
    return x_start <= x <= x_end and y_start <= y <= y_end

# 定义房间信息
ROOMS = {
    'RestRoom': {'color': 'yellow', 'position and size': (0, 3, 7.5, 10.0)},
    'MasterBedroom': {'color': 'green', 'position and size': (7, 10, 5.5, 10.0)},
    'RestRoom2': {'color': 'pink', 'position and size': (7, 10, 2, 4)},
    'ExerciseRoom': {'color': 'deepblue', 'position and size': (4, 6, 8, 10)},
    'LivingRoom': {'color': 'blue', 'position and size': (2, 5, 3, 6)},
    'Kitchen': {'color': 'cyan', 'position and size': (0, 1, 0, 2)},
    'DiningRoom': {'color': 'purple', 'position and size': (2, 4, 0, 1)},
    'Bedroom': {'color': 'red', 'position and size': (5, 10, 0, 2)}
}

def find_closest_waypoint_to_room(trajectory: List[List[float]], room_name: str) -> int:
    """找到轨迹中最接近目标房间的航点索引"""
    room_center = get_room_center(ROOMS[room_name])
    min_dist = float('inf')
    closest_idx = 0
    
    for i, point in enumerate(trajectory):
        dist = calculate_distance(point, room_center)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return closest_idx

def call_deepseek_api(prompt: str, api_key: str, model: str = "deepseek-chat") -> str:
    """
    调用DeepSeek API
    
    Args:
        prompt: 提示词
        api_key: API密钥
        model: 模型名称
    
    Returns:
        API响应内容
    """

    client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-MZT6pRkO220urRvuW4IdSkA3zfXkOAKelUUXqSJgmRUJSIZU",
    base_url = "https://api.chatanywhere.tech/v1"
)
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-v3-2-exp", # gemini-3-pro-preview
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        res = completion.choices[0].message.content
        return res
    except Exception as e:
        print(f"API调用失败: {e}")
        return None

def detect_choice_branch_with_llm(
    trajectory: List[List[float]], 
    choice_constraint: Dict,
    instruction: str,
    api_key: str
) -> Optional[Dict]:
    """
    方法1：使用LLM直接判断轨迹选择的分支
    
    Args:
        trajectory: 轨迹坐标序列
        choice_constraint: choice类型的约束
        instruction: 任务指令
        api_key: DeepSeek API密钥
    
    Returns:
        选择的选项字典
    """
    
    prompt = f"""Given the following information:

Environment:
{json.dumps(ROOMS, indent=2)}

Trajectory:
{json.dumps(trajectory)}

Instruction:
{instruction}

Choice Constraint:
{json.dumps(choice_constraint, indent=2)}

Task: Analyze the trajectory and determine which option from the choice constraint was selected.

Please respond with ONLY a JSON object in this format:
{{
  "selected_option_index": <index>,
  "selected_room": "<room_name>",
  "reasoning": "<brief explanation>"
}}
"""
    
    response = call_deepseek_api(prompt, api_key)
    
    if response:
        try:
            # 提取JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            result = json.loads(response.strip())
            
            # 验证结果
            option_idx = result.get('selected_option_index')
            if option_idx is not None and 0 <= option_idx < len(choice_constraint['options']):
                return choice_constraint['options'][option_idx]
        except:
            pass
    
    return None

def detect_choice_branch_heuristic(
    trajectory: List[List[float]], 
    choice_constraint: Dict,
    previous_time: float = 0
) -> Optional[Dict]:
    """
    方法2：启发式判断轨迹选择的分支（基于距离和房间访问）
    
    Args:
        trajectory: 轨迹坐标序列
        choice_constraint: choice类型的约束
        previous_time: 之前任务的累计时间
    
    Returns:
        选择的选项字典
    """
    
    options = choice_constraint['options']
    
    # 为每个选项计算得分
    option_scores = []
    
    for option in options:
        room_name = option['room']
        score = 0
        
        # 检查轨迹中是否有点接近或进入该房间
        for point in trajectory:
            # 检查是否在房间内
            if is_point_in_room(point, ROOMS[room_name]):
                score += 10  # 在房间内得高分
            else:
                # 检查距离
                room_center = get_room_center(ROOMS[room_name])
                dist = calculate_distance(point, room_center)
                if dist < 2:  # 距离阈值
                    score += 5 - dist  # 越近分越高
        
        option_scores.append({
            'option': option,
            'score': score
        })
    
    # 选择得分最高的选项
    if option_scores:
        best_option = max(option_scores, key=lambda x: x['score'])
        if best_option['score'] > 0:
            return best_option['option']
    
    return None

def assign_timestamps_with_llm(
    trajectory: List[List[float]], 
    stl_constraints: Dict,
    instruction: str,
    api_key: str
) -> List[Tuple[List[float], float]]:
    """
    方法1的完整实现：直接使用LLM为轨迹分配时间戳
    
    Args:
        trajectory: 轨迹坐标序列
        stl_constraints: STL约束
        instruction: 任务指令
        api_key: DeepSeek API密钥
    
    Returns:
        带时间戳的轨迹
    """
    
    prompt = f"""Given the following information:

Environment (rooms with positions):
{json.dumps(ROOMS, indent=2)}

Trajectory (waypoints):
{json.dumps(trajectory)}

Task Instruction:
{instruction}

STL Constraints:
{json.dumps(stl_constraints, indent=2)}

Task: Assign a timestamp to each waypoint in the trajectory based on the constraints.
Consider:
1. The total time bound
2. Sequential constraints and their time bounds
3. Choice constraints and their time bounds
4. Movement speed should be reasonable and consistent

Please respond with ONLY a JSON array of timestamps (one per waypoint):
[<time1>, <time2>, <time3>, ...]

The array must have exactly {len(trajectory)} timestamps.
"""
    
    response = call_deepseek_api(prompt, api_key)
    
    if response:
        try:
            # 提取JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            timestamps = json.loads(response.strip())
            
            # 验证时间戳数量
            if len(timestamps) == len(trajectory):
                return [(trajectory[i], timestamps[i]) for i in range(len(trajectory))]
        except Exception as e:
            print(f"LLM响应解析失败: {e}")
    
    return None

def assign_timestamps_to_trajectory(
    trajectory: List[List[float]], 
    stl_constraints: Dict,
    instruction: str = "",
    api_key: Optional[str] = None,
    use_llm_for_choice: bool = True
) -> List[Tuple[List[float], float]]:
    """
    为轨迹分配时间戳（改进版，支持choice约束）
    
    Args:
        trajectory: 轨迹坐标序列
        stl_constraints: 解析后的STL约束JSON
        instruction: 任务指令
        api_key: DeepSeek API密钥（可选）
        use_llm_for_choice: 是否使用LLM判断choice分支
    
    Returns:
        带时间戳的轨迹
    """
    
    timestamped_trajectory = []
    total_time = stl_constraints.get('time_bounds', {}).get('total', None)
    if isinstance(total_time, list):
        total_time = total_time[-1]
    
    # 如果没有总时间约束，尝试根据距离估算（防止后续计算出错）
    if total_time is None:
        total_distance = sum(
            calculate_distance(trajectory[i], trajectory[i+1]) 
            for i in range(len(trajectory)-1)
        )
        total_time = total_distance if total_distance > 0 else 100.0

    # 处理temporal_constraints
    temporal_constraints = stl_constraints.get('temporal_constraints', [])
    
    if not temporal_constraints:
        # 如果没有时间约束，按距离均匀分配时间
        total_distance = sum(
            calculate_distance(trajectory[i], trajectory[i+1]) 
            for i in range(len(trajectory)-1)
        )
        
        if total_time:
            speed = total_distance / total_time
        else:
            speed = 1.0
        
        current_time = 0
        for i, point in enumerate(trajectory):
            timestamped_trajectory.append((point, current_time))
            if i < len(trajectory) - 1:
                current_time += calculate_distance(point, trajectory[i+1]) / speed
        
        return timestamped_trajectory
    
    # 收集所有任务点（包括sequence和choice）
    all_tasks = []
    
    for constraint in temporal_constraints:
        if constraint['type'] == 'sequence':
            tasks = constraint['tasks']
            for task in tasks:
                if 'target' not in task:
                    continue
                room_name = task['target']
                if not room_name or room_name not in ROOMS:
                    continue
                waypoint_idx = find_closest_waypoint_to_room(trajectory, room_name)
                
                '''# 强制将该航点修正为房间中心，确保空间鲁棒性为正
                room_center = get_room_center(ROOMS[room_name])
                # 注意：这里直接修改了trajectory中的点
                trajectory[waypoint_idx] = [room_center[0], room_center[1]]'''
                
                time_bound = task.get('time_bound', None)
                if isinstance(time_bound, list):
                    time_bound = time_bound[-1]
                if time_bound is None:
                    time_bound = total_time
                actions = task.get('actions', [])
                
                # 处理wait动作
                wait_duration = 0
                for action in actions:
                    if action.get('type') == 'wait':
                        wait_duration = action.get('duration', 0)
                
                all_tasks.append({
                    'index': waypoint_idx,
                    'room': room_name,
                    'time_bound': time_bound,
                    'wait_duration': wait_duration,
                    'type': 'sequence'
                })
        
        elif constraint['type'] == 'choice':
            # 判断选择了哪个分支
            selected_option = None
            
            if use_llm_for_choice and api_key:
                # 方法1：使用LLM判断
                selected_option = detect_choice_branch_with_llm(
                    trajectory, constraint, instruction, api_key
                )
            
            if selected_option is None:
                # 方法2：启发式判断
                previous_time = all_tasks[-1]['time_bound'] if all_tasks else 0
                selected_option = detect_choice_branch_heuristic(
                    trajectory, constraint, previous_time
                )
            
            if selected_option:
                room_name = selected_option['room']
                if not room_name or room_name not in ROOMS:
                    continue
                waypoint_idx = find_closest_waypoint_to_room(trajectory, room_name)
                
                '''# 强制将该航点修正为房间中心，确保空间鲁棒性为正
                room_center = get_room_center(ROOMS[room_name])
                trajectory[waypoint_idx] = [room_center[0], room_center[1]]'''
                
                time_bound = selected_option.get('time_bound', None)
                if isinstance(time_bound, list):
                    time_bound = time_bound[-1]
                if time_bound is None:
                    time_bound = total_time
                actions = selected_option.get('actions', [])
                
                # 处理wait动作
                wait_duration = 0
                for action in actions:
                    if action.get('type') == 'wait':
                        wait_duration = action.get('duration', 0)
                
                all_tasks.append({
                    'index': waypoint_idx,
                    'room': room_name,
                    'time_bound': time_bound,
                    'wait_duration': wait_duration,
                    'type': 'choice'
                })
    
    # 按轨迹顺序排序
    all_tasks.sort(key=lambda x: x['index'])
    
    if not all_tasks:
        # 如果没有有效任务，按距离均匀分配时间
        total_distance = sum(
            calculate_distance(trajectory[i], trajectory[i+1]) 
            for i in range(len(trajectory)-1)
        )
        
        speed = total_distance / total_time if total_time else 1.0
        
        current_time = 0
        for i, point in enumerate(trajectory):
            timestamped_trajectory.append((point, current_time))
            if i < len(trajectory) - 1:
                current_time += calculate_distance(point, trajectory[i+1]) / speed
        
        return timestamped_trajectory
    
    # 填充缺失的时间界限
    last_idx = 0
    last_time = 0
    for task in all_tasks:
        # 计算从上一个任务点到当前任务点的距离
        dist = sum(calculate_distance(trajectory[k], trajectory[k+1]) for k in range(last_idx, task['index']))
        
        if task['time_bound'] is None:
            # 如果没有时间界限，假设速度为1.0进行估算
            # 加上wait_duration，因为time_bound通常指任务完成时间（包含等待）
            task['time_bound'] = last_time + dist + task['wait_duration']
        
        # 更新last_idx和last_time
        last_idx = task['index']
        last_time = task['time_bound']
    
    # 创建任务索引映射，方便查找
    task_map = {t['index']: t for t in all_tasks}

    # 分配时间戳
    for i, point in enumerate(trajectory):
        # 找到当前点属于哪个任务段
        current_segment = None
        for j, task in enumerate(all_tasks):
            if i <= task['index']:
                current_segment = j
                break
        
        if current_segment is None:
            current_segment = len(all_tasks) - 1
        
        # 计算时间
        if current_segment == 0:
            # 第一段：从起点到第一个任务点
            start_idx = 0
            end_idx = all_tasks[0]['index']
            start_time = 0
            # 终点时间应该是到达时间 = 完成时间 - 等待时间
            end_time = all_tasks[0]['time_bound'] - all_tasks[0]['wait_duration']
        else:
            # 后续段
            start_idx = all_tasks[current_segment - 1]['index']
            end_idx = all_tasks[current_segment]['index']
            # 起点时间是上一个任务的完成时间（已包含等待）
            start_time = all_tasks[current_segment - 1]['time_bound']
            # 终点时间是当前任务的到达时间
            end_time = all_tasks[current_segment]['time_bound'] - all_tasks[current_segment]['wait_duration']
        
        # 线性插值计算当前时间
        if end_idx == start_idx:
            current_time = end_time
        else:
            # 计算段内距离
            segment_distance = sum(
                calculate_distance(trajectory[k], trajectory[k+1])
                for k in range(start_idx, end_idx)
            )
            
            if segment_distance == 0:
                ratio = 0
            else:
                current_distance = sum(
                    calculate_distance(trajectory[k], trajectory[k+1])
                    for k in range(start_idx, min(i, end_idx))
                )
                ratio = current_distance / segment_distance
            
            current_time = start_time + (end_time - start_time) * ratio
        
        timestamped_trajectory.append((point, round(current_time, 2)))

        # 如果当前点是任务点且有等待时间，插入等待结束的点
        if i in task_map and task_map[i]['wait_duration'] > 0:
            departure_time = current_time + task_map[i]['wait_duration']
            # 插入同一个点，但时间增加
            timestamped_trajectory.append((point, round(departure_time, 2)))
    
    return timestamped_trajectory

def process_trajectory_data(
    data_entry: Dict, 
    api_key: Optional[str] = None,
    method: str = "hybrid"  # "hybrid", "llm_full", "heuristic"
) -> Dict:
    """
    处理单个轨迹数据条目
    
    Args:
        data_entry: 数据条目
        api_key: DeepSeek API密钥
        method: 方法选择
            - "hybrid": choice用启发式或LLM，时间分配用插值（推荐）
            - "llm_full": 完全使用LLM分配时间戳
            - "heuristic": 完全使用启发式方法
    
    Returns:
        处理结果
    """
    
    # 解析STL JSON
    stl_constraints = parse_stl_json(data_entry['stl_json'])
    
    # 解析轨迹
    target = data_entry['target']
    if isinstance(target, list):
        trajectory = target
    else:
        trajectory = eval(target)
    
    instruction = data_entry.get('instruction', '')
    
    # 根据方法选择
    if method == "llm_full" and api_key:
        # 方法1：完全使用LLM
        timestamped_trajectory = assign_timestamps_with_llm(
            trajectory, stl_constraints, instruction, api_key
        )
        
        # 如果LLM失败，fallback到混合方法
        if timestamped_trajectory is None:
            print("LLM方法失败，使用混合方法")
            timestamped_trajectory = assign_timestamps_to_trajectory(
                trajectory, stl_constraints, instruction, api_key, 
                use_llm_for_choice=True
            )
    
    elif method == "heuristic":
        # 方法2：完全使用启发式
        timestamped_trajectory = assign_timestamps_to_trajectory(
            trajectory, stl_constraints, instruction, None, 
            use_llm_for_choice=False
        )
    
    else:  # hybrid (推荐)
        # 混合方法：choice用LLM判断（如果有API key），时间分配用插值
        timestamped_trajectory = assign_timestamps_to_trajectory(
            trajectory, stl_constraints, instruction, api_key, 
            use_llm_for_choice=(api_key is not None)
        )
    
    return {
        'instruction': instruction,
        'trajectory': trajectory,
        'timestamped_trajectory': timestamped_trajectory,
        'constraints': stl_constraints,
        'method_used': method
    }

# 测试示例
if __name__ == "__main__":
    import os
    
    # 输入和输出文件路径
    input_file = "/home/lijia/code/1113_CLHS/train_dataset/train_low_with_stl_output_with_target.json"
    output_file = "/home/lijia/code/1113_CLHS/train_dataset/train_low_with_stl_output_with_target_and_pwl.json"
    
    # 从环境变量获取API key（如果有）
    api_key = "sk-MZT6pRkO220urRvuW4IdSkA3zfXkOAKelUUXqSJgmRUJSIZU"
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"输入文件不存在: {input_file}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        exit(1)
    
    if not isinstance(data, list):
        print("输入文件不是列表格式")
        exit(1)
    
    processed_data = []
    total_items = len(data)
    print(f"开始处理 {total_items} 个条目...")
    
    for i, item in enumerate(data):
        #if (i + 1) % 10 == 0:
        print(f"正在处理第 {i + 1}/{total_items} 个条目...")

        if not isinstance(item, dict):
            print(f"跳过非dict项: {item}")
            continue
        
        # 复制原有字段
        new_item = item.copy()
        
        # 调用混合方法处理轨迹
        try:
            result = process_trajectory_data(item, api_key, method="hybrid")
            timestamped_trajectory = result['timestamped_trajectory']
            
            # 格式化为[[x1,y1,t1],[x2,y2,t2],...]
            pwl = [[point[0], point[1], time] for point, time in timestamped_trajectory]
            new_item['pwl'] = pwl
        except Exception as e:
            print(f"处理轨迹失败，跳过此项: {e}, 指令: {item.get('instruction', 'unknown')}")
            continue
        
        processed_data.append(new_item)
    
    # 保存到新文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"处理完成，共处理 {len(processed_data)} 个条目，保存到 {output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        exit(1)