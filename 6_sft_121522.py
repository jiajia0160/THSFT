import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json
import re
import numpy as np
from stlcgpp.formula import *
import os
from typing import Dict, List, Optional, Tuple
import warnings
from tqdm import tqdm
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings('ignore')

'''
自训练流程：
阶段1（预训练回归头，2-3epochs）：
Freeze LLM（只做forward获取hidden states）
用ground truth的<result>
训练MLP：MSE loss （预测轨迹 vs ground truth)
不更新LLM参数
阶段2（微调LLM，5-10epochs）：
Freeze MLP（固定回归头）
训练LLM生成正确的<result>
Loss = CE loss + STL loss（通过固定的MLP计算）
STL梯度会反传到LLM，引导它生成满足约束的轨迹
'''

# -----------------------------------------------------------------------------
# 1. 保持之前的辅助函数和类
# -----------------------------------------------------------------------------

def differentiable_interpolate(keypoints: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
    """Differentiable linear interpolation."""
    B, N, _ = keypoints.shape
    t = keypoints[:, :, 2]
    sorted_indices = torch.argsort(t, dim=1)
    gather_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
    keypoints_sorted = torch.gather(keypoints, 1, gather_indices)
    
    x = keypoints_sorted[:, :, 0]
    y = keypoints_sorted[:, :, 1]
    t = keypoints_sorted[:, :, 2]
    
    if t_query.dim() == 1:
        t_query = t_query.unsqueeze(0).expand(B, -1)
    
    M = t_query.shape[1]
    indices = torch.searchsorted(t, t_query, right=True) - 1
    indices = indices.clamp(0, N - 2)
    
    def gather_val(input_tensor, idx):
        return torch.gather(input_tensor, 1, idx)
    
    t0 = gather_val(t, indices)
    t1 = gather_val(t, indices + 1)
    x0 = gather_val(x, indices)
    x1 = gather_val(x, indices + 1)
    y0 = gather_val(y, indices)
    y1 = gather_val(y, indices + 1)
    
    dt = t1 - t0
    eps = 1e-8
    dt = torch.where(dt.abs() < eps, torch.ones_like(dt) * eps, dt)
    w = ((t_query - t0) / dt).clamp(0.0, 1.0)
    
    x_interp = x0 + w * (x1 - x0)
    y_interp = y0 + w * (y1 - y0)
    
    return torch.stack([x_interp, y_interp], dim=-1)


class DifferentiableSTLConverter:
    """STL Converter - 使用之前的完整实现"""
    def __init__(self, rooms_dict: Dict):
        self.rooms = rooms_dict
        
    def get_room_name(self, item: Dict) -> Optional[str]:
        for key in ['target', 'room', 'area', 'location', 'nearest_room']:
            if key in item and item[key] is not None:
                return item[key]
        return None

    def create_room_predicate(self, room_name: str):
        if room_name not in self.rooms:
            return None
        x_min, x_max, y_min, y_max = self.rooms[room_name]
        
        def in_room(states):
            x, y = states[..., 0], states[..., 1]
            dx_min = x - x_min
            dx_max = x_max - x
            dy_min = y - y_min
            dy_max = y_max - y
            robustness = torch.min(torch.min(dx_min, dx_max), torch.min(dy_min, dy_max))
            return robustness + 1e-6
        
        return Predicate(f"in_{room_name}", predicate_function=in_room)
    
    def time_to_steps(self, time_bound, dt=0.1):
        if time_bound is None:
            return None
        if isinstance(time_bound, list):
            return [int(t / dt) for t in time_bound]
        return int(time_bound / dt)
    
    def build_stl_formula(self, stl_json: Dict, dt: float = 0.1):
        formulas = []
        if 'temporal_constraints' in stl_json:
            for constraint in stl_json['temporal_constraints']:
                try:
                    formula = self._build_constraint(constraint, dt)
                    if formula:
                        formulas.append(formula)
                except:
                    pass
        
        if 'global_constraints' in stl_json:
            for constraint in stl_json['global_constraints']:
                try:
                    formula = self._build_global_constraint(constraint, dt)
                    if formula:
                        formulas.append(formula)
                except:
                    pass
        
        if len(formulas) == 0:
            return None
        elif len(formulas) == 1:
            return formulas[0]
        else:
            combined = formulas[0]
            for f in formulas[1:]:
                combined = combined & f
            return combined
    
    def _build_constraint(self, constraint: Dict, dt: float):
        if constraint['type'] == 'sequence':
            return self._build_sequence(constraint, dt)
        elif constraint['type'] == 'choice':
            return self._build_choice(constraint, dt)
        return None
    
    def _build_sequence(self, constraint: Dict, dt: float):
        tasks = constraint['tasks']
        absolute_constraints = []
        
        for task in tasks:
            room_name = self.get_room_name(task)
            if not room_name:
                continue
            room_pred = self.create_room_predicate(room_name)
            if room_pred is None:
                continue
            
            base_pred = room_pred > 0
            
            if 'actions' in task:
                for action in task['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])
            
            if 'time_bound' in task and task['time_bound']:
                time_bound = task['time_bound']
                if isinstance(time_bound, list):
                    interval = self.time_to_steps(time_bound, dt)
                    absolute_constraints.append(Eventually(base_pred, interval=interval))
                else:
                    steps = self.time_to_steps(time_bound, dt)
                    absolute_constraints.append(Eventually(base_pred, interval=[0, steps + 1]))
            else:
                absolute_constraints.append(Eventually(base_pred))
        
        ordering_formula = None
        for i in range(len(tasks) - 1, -1, -1):
            task = tasks[i]
            room_name = self.get_room_name(task)
            if not room_name:
                continue
            room_pred = self.create_room_predicate(room_name)
            if room_pred is None:
                continue
            
            base_pred = room_pred > 0
            
            if 'actions' in task:
                for action in task['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])
            
            if ordering_formula is None:
                ordering_formula = Eventually(base_pred)
            else:
                ordering_formula = Eventually(base_pred & ordering_formula)
        
        if ordering_formula is None:
            return None
        
        final_formula = ordering_formula
        for c in absolute_constraints:
            final_formula = final_formula & c
        return final_formula

    def _build_choice(self, constraint: Dict, dt: float):
        options = constraint['options']
        formulas = []
        
        for option in options:
            room_name = self.get_room_name(option)
            if not room_name:
                continue
            room_pred = self.create_room_predicate(room_name)
            if room_pred is None:
                continue
            
            base_pred = room_pred > 0
            
            if 'actions' in option:
                for action in option['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])
            
            if 'time_bound' in option and option['time_bound']:
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
        
        if len(formulas) == 0:
            return None
        elif len(formulas) == 1:
            return formulas[0]
        else:
            combined = formulas[0]
            for f in formulas[1:]:
                combined = combined | f
            return combined

    def _build_global_constraint(self, constraint: Dict, dt: float):
        return None


# -----------------------------------------------------------------------------
# 2. Dataset (使用之前的实现)
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = """I will provide the position and size of each box in the environment, a natural language instruction to enter or avoid boxes.

The position and size of each box are described by the coordinates: (x_start, x_end, y_start, y_end).

The size of the environmental site is (0,10,0,10). 
Scene objects:
['name': 'RestRoom', 'color': 'yellow', 'position and size': (0, 3, 7.5, 10.0)]
['name': 'MasterBedroom', 'color': 'green', 'position and size': (7, 10, 5.5, 10.0)]
['name': 'RestRoom2', 'color': 'pink', 'position and size': (7, 10, 2, 4)]
['name': 'ExerciseRoom', 'color': 'deepblue', 'position and size': (4, 6, 8, 10)]
['name': 'LivingRoom', 'color': 'blue', 'position and size': (2, 5, 3, 6)]
['name': 'Kitchen', 'color': 'cyan', 'position and size': (0, 1, 0, 2)]
['name': 'DiningRoom', 'color': 'purple', 'position and size': (2, 4, 0, 1)]
['name': 'Bedroom', 'color': 'red', 'position and size': (5, 10, 0, 2)]

The output trajectory is represented in format:
[[x1,y1,t1],[x2,y2,t2],...]
Here, [x, y, t] indicates that the robot is at the position [x, y] at time t.

Output format:
<result>[[x1,y1,t1],[x2,y2,t2],...]</result>
<reasoning>Your step-by-step reasoning</reasoning>

Input:
The initial position of the robot is [1,3].
instruction: {instruction}
"""


class RobotTrajectoryDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int = 1024, 
                 max_traj_len: int = 30):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_traj_len = max_traj_len
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self._clean_data()
    
    def _clean_data(self):
        for item in self.data:
            output = item['output']
            output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
            output = re.sub(r'\s+', ' ', output).strip()
            item['output'] = output
    
    def _extract_result_content(self, text: str) -> Optional[str]:
        match = re.search(r'<result>(.*?)</result>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _find_result_token_positions(self, input_ids: torch.Tensor, 
                                    result_text: str) -> Tuple[int, int]:
        """找到<result>内容在token序列中的位置"""
        result_tokens = self.tokenizer.encode(result_text, add_special_tokens=False)
        input_list = input_ids.tolist()
        
        # 精确匹配
        for i in range(len(input_list) - len(result_tokens) + 1):
            if input_list[i:i+len(result_tokens)] == result_tokens:
                return i, i + len(result_tokens)
        
        # 标签匹配
        result_start_token = self.tokenizer.encode("<result>", add_special_tokens=False)
        result_end_token = self.tokenizer.encode("</result>", add_special_tokens=False)
        
        start_pos = None
        end_pos = None
        
        for i in range(len(input_list)):
            if input_list[i:i+len(result_start_token)] == result_start_token:
                start_pos = i + len(result_start_token)
            if input_list[i:i+len(result_end_token)] == result_end_token:
                end_pos = i
                break
        
        if start_pos is not None and end_pos is not None and start_pos < end_pos:
            return start_pos, end_pos
        
        # Fallback
        seq_len = len(input_list)
        return int(seq_len * 0.8), seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Trajectory
        pwl = item['pwl']
        pwl_tensor = torch.tensor(pwl, dtype=torch.float32)
        
        if pwl_tensor.shape[0] > self.max_traj_len:
            pwl_tensor = pwl_tensor[:self.max_traj_len]
        else:
            padding = torch.zeros(self.max_traj_len - pwl_tensor.shape[0], 3)
            pwl_tensor = torch.cat([pwl_tensor, padding], dim=0)
        
        # Initial position
        if len(pwl) > 0:
            init_x, init_y = pwl[0][0], pwl[0][1]
            initial_pos_str = f"[{init_x},{init_y}]"
        else:
            initial_pos_str = "[1,3]"
        
        instruction_raw = item['instruction']
        output_raw = item['output']
        
        instruction = PROMPT_TEMPLATE.format(
            initial_pos=initial_pos_str,
            instruction=instruction_raw
        )
        
        result_content = self._extract_result_content(output_raw)
        if result_content is None:
            result_content = str(pwl)
        
        # --- Smart Truncation Logic ---
        # 1. Construct full text to check length
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output_raw}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(full_tokens) > self.max_length:
            # Need truncation. Strategy: Keep instruction and result, truncate reasoning middle.
            
            # Extract parts
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', output_raw, re.DOTALL)
            if reasoning_match:
                reasoning_content = reasoning_match.group(1)
                
                # Construct a "skeleton" without the reasoning body to see how much space we have
                # Skeleton: Instruction + <reasoning>...truncated...</reasoning> + <result>...</result>
                # We use a placeholder for reasoning to calculate overhead
                placeholder = " ... truncated ... "
                skeleton_output = output_raw.replace(reasoning_content, placeholder)
                
                skeleton_messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": skeleton_output}
                ]
                skeleton_text = self.tokenizer.apply_chat_template(
                    skeleton_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                skeleton_tokens = self.tokenizer.encode(skeleton_text, add_special_tokens=False)
                
                available_tokens = self.max_length - len(skeleton_tokens)
                
                if available_tokens > 0:
                    # We have space for some reasoning. 
                    # Let's keep start and end of reasoning.
                    # Approximate char/token ratio (conservative estimate, e.g. 3 chars per token)
                    # Better: encode reasoning and slice tokens
                    reasoning_tokens = self.tokenizer.encode(reasoning_content, add_special_tokens=False)
                    
                    if len(reasoning_tokens) > available_tokens:
                        # Keep start and end
                        keep_len = available_tokens // 2
                        start_tokens = reasoning_tokens[:keep_len]
                        end_tokens = reasoning_tokens[-keep_len:]
                        
                        new_reasoning = self.tokenizer.decode(start_tokens) + " ... [truncated] ... " + self.tokenizer.decode(end_tokens)
                        output_raw = output_raw.replace(reasoning_content, new_reasoning)
                    # else: reasoning fits, weird that full_tokens > max_length, maybe overhead calculation was off?
                    # In that case, standard truncation will handle it, but we tried our best.
                else:
                    # No space for reasoning at all, just keep placeholder
                    output_raw = skeleton_output
            
            # Re-construct messages with potentially truncated output
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output_raw}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

        # Standard encoding (now likely within limits or close to it)
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        labels = input_ids.clone()
        
        # Mask instruction
        user_messages = [{"role": "user", "content": instruction}]
        user_text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        user_enc = self.tokenizer(user_text, return_tensors='pt')
        instr_len = user_enc['input_ids'].shape[1]
        
        if instr_len < self.max_length:
            labels[:instr_len] = -100
        labels[attention_mask == 0] = -100
        
        # Find <result> position
        result_start, result_end = self._find_result_token_positions(input_ids, result_content)
        
        result_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        result_mask[result_start:result_end] = True
        
        # STL JSON
        stl_json_str = item.get('stl_json', '{}')
        stl_json_str = re.sub(r'^```json\s*', '', stl_json_str)
        stl_json_str = re.sub(r'\s*```$', '', stl_json_str)
        try:
            stl_json = json.loads(stl_json_str)
        except:
            stl_json = {}
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pwl': pwl_tensor,
            'result_mask': result_mask,
            'stl_json': stl_json,
            'result_text': result_content
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'pwl': torch.stack([item['pwl'] for item in batch]),
        'result_masks': torch.stack([item['result_mask'] for item in batch]),
        'stl_jsons': [item['stl_json'] for item in batch],
        'result_texts': [item['result_text'] for item in batch]
    }


# -----------------------------------------------------------------------------
# 3. 模型架构
# -----------------------------------------------------------------------------

class MaskedTrajectoryDecoder(nn.Module):
    """轨迹解码器：从<result> tokens的hidden states解码轨迹"""
    def __init__(self, hidden_size: int, max_traj_len: int = 30):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_traj_len = max_traj_len
        
        # Token-level attention
        self.token_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Spatial decoder
        self.spatial_decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, max_traj_len * 2)
        )
        
        # Time decoder
        self.time_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_traj_len),
            nn.Softplus()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor, result_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, H]
            result_mask: [B, L] boolean
        Returns:
            traj_pred: [B, max_traj_len, 3]
        """
        batch_size = hidden_states.shape[0]
        
        # Extract result region
        mask_expanded = result_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask_expanded
        
        # Attention over result tokens
        attn_logits = self.token_attention(masked_hidden)
        attn_logits = attn_logits.masked_fill(~result_mask.unsqueeze(-1), float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # Weighted aggregation
        attended_features = (masked_hidden * attn_weights).sum(dim=1)
        
        # Transform
        features = self.feature_transform(attended_features)
        
        # Decode spatial
        spatial = self.spatial_decoder(features)
        spatial = spatial.view(batch_size, self.max_traj_len, 2)
        
        # Decode time with monotonicity
        time_intervals = self.time_decoder(features)
        time_monotonic = torch.cumsum(time_intervals, dim=1)
        
        # Combine
        traj_pred = torch.cat([spatial, time_monotonic.unsqueeze(-1)], dim=-1)
        
        return traj_pred


class MultimodalRobotModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        max_traj_len: int = 30,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        gradient_checkpointing: bool = False,
        device_map: Optional[str] = None,
    ):
        super().__init__()

        self.use_lora = use_lora
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False  # 训练时必须关闭cache，否则显存会爆炸

        if self.use_lora:
            target_modules = lora_target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.llm = get_peft_model(self.llm, lora_config)

        self.hidden_size = self.llm.config.hidden_size
        self.max_traj_len = max_traj_len
        
        self.trajectory_decoder = MaskedTrajectoryDecoder(
            hidden_size=self.hidden_size,
            max_traj_len=max_traj_len
        )
        
        # Cast decoder to bfloat16
        self.trajectory_decoder.to(torch.bfloat16)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                result_masks: Optional[torch.Tensor] = None,
                return_hidden_states: bool = True):
        
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=return_hidden_states,
            return_dict=True
        )
        
        loss_ce = outputs.loss if labels is not None else None
        hidden_states = outputs.hidden_states[-1] if return_hidden_states else None
        
        traj_pred = None
        if result_masks is not None and hidden_states is not None:
            # Ensure hidden_states and result_masks are on the same device as decoder
            decoder_device = self.trajectory_decoder.spatial_decoder[0].weight.device
            hidden_states = hidden_states.to(decoder_device)
            result_masks = result_masks.to(decoder_device)
            traj_pred = self.trajectory_decoder(hidden_states, result_masks)
        
        return {
            'loss_ce': loss_ce,
            'logits': outputs.logits,
            'traj_pred': traj_pred,
            'hidden_states': hidden_states
        }


# -----------------------------------------------------------------------------
# 4. 阶段1：预训练回归头（Freeze LLM）
# -----------------------------------------------------------------------------

def pretrain_decoder_epoch(model: nn.Module, dataloader: DataLoader, 
                          optimizer, device: torch.device, epoch: int):
    """
    阶段1：只训练MLP解码器
    LLM被freeze，只用来提取hidden states
    """
    model.train()
    # Freeze LLM
    for param in model.llm.parameters():
        param.requires_grad = False
    
    # Unfreeze decoder
    for param in model.trajectory_decoder.parameters():
        param.requires_grad = True
    
    total_mse_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pwl_gt = batch['pwl'].to(device)
        result_masks = batch['result_masks'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Forward (no gradient on LLM)
            with torch.no_grad():
                llm_outputs = model.llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = llm_outputs.hidden_states[-1]
            
            # Decoder forward (with gradient)
            traj_pred = model.trajectory_decoder(hidden_states, result_masks)
            
            # MSE loss only on valid trajectory points
            valid_mask = (pwl_gt.sum(dim=-1) != 0).float().unsqueeze(-1)
            mse_loss = torch.sum(((traj_pred - pwl_gt) ** 2) * valid_mask)
            mse_loss = mse_loss / (valid_mask.sum() + 1e-8)
        
        # Backward (only update decoder)
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trajectory_decoder.parameters(), 1.0)
        optimizer.step()
        
        total_mse_loss += mse_loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'MSE': f'{mse_loss.item():.4f}'})
    
    avg_mse = total_mse_loss / num_batches
    print(f"Stage 1 Epoch {epoch} - Average MSE: {avg_mse:.4f}")
    return avg_mse


# -----------------------------------------------------------------------------
# 5. 阶段2：微调LLM（Freeze Decoder）
# -----------------------------------------------------------------------------

def compute_stl_loss(traj_pred: torch.Tensor, stl_jsons: List[Dict],
                    stl_converter, device: torch.device) -> torch.Tensor:
    """计算STL约束损失"""
    batch_size = traj_pred.shape[0]
    total_stl_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
    valid_count = 0
    
    for i in range(batch_size):
        if not stl_jsons[i]:
            continue
        
        try:
            formula = stl_converter.build_stl_formula(stl_jsons[i], dt=0.1)
            if formula is None:
                continue
            
            kp = traj_pred[i:i+1]
            kp[:, :, 2] = torch.abs(kp[:, :, 2])
            kp[:, :, 2], _ = torch.sort(kp[:, :, 2], dim=1)
            
            t_max = max(kp[0, :, 2].max().item(), 1.0)
            # 降低采样点数以节省显存 (100 -> 50)
            t_query = torch.linspace(0, t_max, 80, device=device)
            
            states = differentiable_interpolate(kp, t_query).squeeze(0)
            robustness = formula.robustness(states, 0)
            
            if not (torch.isnan(robustness) or torch.isinf(robustness)):
                stl_loss_sample = torch.relu(-robustness + 0.1)
                stl_loss_sample = torch.clamp(stl_loss_sample, max=10.0)
                total_stl_loss += stl_loss_sample
                valid_count += 1
        except:
            continue
    
    return total_stl_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=device, dtype=torch.bfloat16)


def finetune_llm_epoch(model: nn.Module, dataloader: DataLoader, 
                      optimizer, scaler, scheduler, stl_converter,
                      device: torch.device, epoch: int,
                      lambda_stl: float = 0.1,
                      lambda_ce: float = 1.0,
                      max_grad_norm: float = 1.0):
    """阶段2：冻结轨迹解码器，只训练LLM。

    Loss = lambda_ce * CE + lambda_stl * STL
    STL loss 通过固定解码器从<result> hidden states解码轨迹，梯度回传到LLM。
    """
    model.train()

    # Freeze decoder
    for param in model.trajectory_decoder.parameters():
        param.requires_grad = False

    # Unfreeze LLM
    for param in model.llm.parameters():
        param.requires_grad = True

    total_ce_loss = 0.0
    total_stl_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        result_masks = batch['result_masks'].to(device)
        stl_jsons = batch['stl_jsons']

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                result_masks=result_masks,
                return_hidden_states=True
            )

            loss_ce = outputs['loss_ce']
            traj_pred = outputs['traj_pred']

            # CE loss can be float32 internally; keep as float32 for stability
            loss_ce_f = loss_ce.float() if loss_ce is not None else torch.tensor(0.0, device=device)

            # STL loss computed in bf16; cast to float32 for sum
            if traj_pred is not None:
                loss_stl = compute_stl_loss(traj_pred, stl_jsons, stl_converter, device)
            else:
                loss_stl = torch.tensor(0.0, device=device, dtype=torch.bfloat16)

            loss_stl_f = loss_stl.float()
            loss = lambda_ce * loss_ce_f + lambda_stl * loss_stl_f

        if torch.isnan(loss) or torch.isinf(loss):
            progress_bar.set_postfix({'loss': 'nan/inf'})
            continue

        scaler.scale(loss).backward()

        # Only clip grads of trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if len(trainable_params) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_ce_loss += loss_ce_f.item()
        total_stl_loss += loss_stl_f.item()
        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{loss_ce_f.item():.4f}',
            'stl': f'{loss_stl_f.item():.4f}'
        })
        
        # Cleanup to save memory
        del outputs, loss, loss_ce, traj_pred, loss_ce_f, loss_stl, loss_stl_f
        
        # 定期清理显存
        if num_batches % 50 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()

    if num_batches == 0:
        return {
            'loss': float('nan'),
            'loss_ce': float('nan'),
            'loss_stl': float('nan')
        }

    return {
        'loss': total_loss / num_batches,
        'loss_ce': total_ce_loss / num_batches,
        'loss_stl': total_stl_loss / num_batches
    }


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_two_stage(
    model_path: str,
    data_path: str,
    output_dir: str,
    stage1_epochs: int = 300,
    stage2_epochs: int = 10,
    batch_size: int = 1,
    max_length: int = 1800,
    max_traj_len: int = 30,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-5,
    lambda_stl: float = 0.1,
    lambda_ce: float = 1.0,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    num_workers: int = 0,
    resume_path: Optional[str] = None,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    gradient_checkpointing: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    _set_seed(seed)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = RobotTrajectoryDataset(
        json_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_traj_len=max_traj_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    model = MultimodalRobotModel(
        model_path=model_path,
        max_traj_len=max_traj_len,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        gradient_checkpointing=gradient_checkpointing,
        device_map="auto",  # 减少GPU 0的占用，留给activations和decoder，设置"balanced_low_0"
    )
    # model.to(device) # Do not move the whole model when using device_map="auto"
    
    # Use the device of the LLM (usually cuda:0) as the main device for inputs and decoder
    if hasattr(model.llm, "device"):
        device = model.llm.device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.trajectory_decoder.to(device)

    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        if 'llm' in ckpt:
            model.llm.load_state_dict(ckpt['llm'], strict=False)
        if 'trajectory_decoder' in ckpt:
            model.trajectory_decoder.load_state_dict(ckpt['trajectory_decoder'], strict=False)
        print(f"Resumed from checkpoint: {resume_path}")

    stl_converter = DifferentiableSTLConverter(ROOMS)
    #scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    # --- Log Experiment Config ---
    config_log_path = os.path.join(output_dir, 'experiment_log.txt')
    with open(config_log_path, 'w') as f:
        f.write("=== Experiment Configuration ===\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Output Dir: {output_dir}\n")
        f.write(f"Stage 1 Epochs: {stage1_epochs}\n")
        f.write(f"Stage 2 Epochs: {stage2_epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Max Length: {max_length}\n")
        f.write(f"Max Traj Len: {max_traj_len}\n")
        f.write(f"LR Stage 1: {lr_stage1}\n")
        f.write(f"LR Stage 2: {lr_stage2}\n")
        f.write(f"Lambda STL: {lambda_stl}\n")
        f.write(f"Lambda CE: {lambda_ce}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Warmup Ratio: {warmup_ratio}\n")
        f.write(f"Max Grad Norm: {max_grad_norm}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Use LoRA: {use_lora}\n")
        if use_lora:
            f.write(f"LoRA R: {lora_r}\n")
            f.write(f"LoRA Alpha: {lora_alpha}\n")
            f.write(f"LoRA Dropout: {lora_dropout}\n")
            f.write(f"LoRA Target Modules: {lora_target_modules}\n")
        f.write(f"Gradient Checkpointing: {gradient_checkpointing}\n")
        f.write(f"Device: {device}\n")
        f.write("================================\n\n")

    # -------------------------
    # Stage 1: pretrain decoder
    # -------------------------
    print(
        f"Config: device={device}, bs={batch_size}, max_len={max_length}, max_traj_len={max_traj_len}, "
        f"stage1_epochs={stage1_epochs}, stage2_epochs={stage2_epochs}, lr1={lr_stage1}, lr2={lr_stage2}, "
        f"lambda_ce={lambda_ce}, lambda_stl={lambda_stl}, use_lora={use_lora}, lora_r={lora_r}, "
        f"lora_alpha={lora_alpha}, lora_dropout={lora_dropout}, grad_ckpt={gradient_checkpointing}"
    )
    if stage1_epochs > 0:
        decoder_params = list(model.trajectory_decoder.parameters())
        opt1 = torch.optim.AdamW(decoder_params, lr=lr_stage1, weight_decay=weight_decay)

        with open(config_log_path, 'a') as f:
            f.write("=== Stage 1 Training ===\n")

        for epoch in range(1, stage1_epochs + 1):
            avg_mse = pretrain_decoder_epoch(model, dataloader, opt1, device, epoch)
            with open(config_log_path, 'a') as f:
                f.write(f"Epoch {epoch}: MSE={avg_mse:.4f}\n")

        ckpt_path = os.path.join(output_dir, 'stage1_decoder.pt')
        torch.save({'trajectory_decoder': model.trajectory_decoder.state_dict()}, ckpt_path)
        print(f"Saved stage1 decoder to: {ckpt_path}")

    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------
    # Stage 2: finetune LLM
    # -------------------------
    if stage2_epochs > 0:
        from transformers import get_linear_schedule_with_warmup
        
        with open(config_log_path, 'a') as f:
            f.write("\n=== Stage 2 Training ===\n")

        # Configure trainable params: LoRA-only if enabled, otherwise full LLM
        for param in model.trajectory_decoder.parameters():
            param.requires_grad = False

        if model.use_lora:
            # Freeze base weights; enable only LoRA adapters
            for name, param in model.llm.named_parameters():
                param.requires_grad = "lora_" in name
        else:
            for param in model.llm.parameters():
                param.requires_grad = True

        llm_params = [p for p in model.llm.parameters() if p.requires_grad]
        opt2 = torch.optim.AdamW(llm_params, lr=lr_stage2, weight_decay=weight_decay)

        total_steps = max(len(dataloader) * stage2_epochs, 1)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(opt2, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_loss = float('inf')
        for epoch in range(1, stage2_epochs + 1):
            metrics = finetune_llm_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=opt2,
                scaler=scaler,
                scheduler=scheduler,
                stl_converter=stl_converter,
                device=device,
                epoch=epoch,
                lambda_stl=lambda_stl,
                lambda_ce=lambda_ce,
                max_grad_norm=max_grad_norm
            )
            print(f"Stage 2 Epoch {epoch} - loss={metrics['loss']:.4f}, ce={metrics['loss_ce']:.4f}, stl={metrics['loss_stl']:.4f}")
            
            with open(config_log_path, 'a') as f:
                f.write(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, CE={metrics['loss_ce']:.4f}, STL={metrics['loss_stl']:.4f}\n")

            ckpt = {
                'llm': model.llm.state_dict(),
                'trajectory_decoder': model.trajectory_decoder.state_dict(),
                'tokenizer_name_or_path': model_path,
                'epoch': epoch,
                'metrics': metrics
            }
            last_path = os.path.join(output_dir, 'stage2_last.pt')
            torch.save(ckpt, last_path)

            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best_path = os.path.join(output_dir, 'stage2_best.pt')
                torch.save(ckpt, best_path)
                print(f"Saved best checkpoint to: {best_path}")



def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(description='Two-stage self-training for robot trajectory with STL.')
    p.add_argument('--model_path', type=str, default='/home/lijia/code/LLaMA-Factory/models/Qwen/Qwen3-1.7B')
    p.add_argument('--data_path', type=str, default='/home/lijia/code/1113_CLHS/1209_git/THSFT/positive_robustness.json')
    p.add_argument('--output_dir', type=str, default='/home/lijia/code/1208_CLHS/outputs_6_sft_121724')

    p.add_argument('--stage1_epochs', type=int, default=300)
    p.add_argument('--stage2_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--max_length', type=int, default=1800)  # 约束的是 指令 + 真值输出 + chat模板 的整体 token 数。512token可能不够
    p.add_argument('--max_traj_len', type=int, default=30)
    p.add_argument('--lr_stage1', type=float, default=1e-3)
    p.add_argument('--lr_stage2', type=float, default=1e-5)
    p.add_argument('--lambda_stl', type=float, default=0.1)
    p.add_argument('--lambda_ce', type=float, default=1.0)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--resume_path', type=str, default=None)
    p.add_argument('--use_lora', action='store_true', default=True,
                   help='Use LoRA adapters for stage 2 finetuning (default: True).')
    p.add_argument('--no_lora', action='store_false', dest='use_lora',
                   help='Disable LoRA; train full model.')
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_target_modules', type=str, nargs='+', default=None,
                   help='Target module names for LoRA (space separated).')
    p.add_argument('--gradient_checkpointing', action='store_true', default=True,
                   help='Enable gradient checkpointing to save memory in stage 2.')
    return p


if __name__ == '__main__':
    args = _build_argparser().parse_args()
    train_two_stage(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_traj_len=args.max_traj_len,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        lambda_stl=args.lambda_stl,
        lambda_ce=args.lambda_ce,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        num_workers=args.num_workers,
        resume_path=args.resume_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        gradient_checkpointing=args.gradient_checkpointing,
    )