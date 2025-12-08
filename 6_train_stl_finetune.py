import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import json
import re
import numpy as np
from stlcgpp.formula import *
import os

# 设置环境变量以避免一些并行问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# 1. Differentiable Interpolation and STL Converter
# -----------------------------------------------------------------------------

def differentiable_interpolate(keypoints, t_query):
    """
    Differentiable linear interpolation.
    
    Args:
        keypoints: Tensor of shape (Batch, N, 3) where last dim is (x, y, t).
                   Assumes t is strictly increasing for each batch item.
        t_query: Tensor of shape (M,) or (Batch, M). The times to evaluate at.
        
    Returns:
        Tensor of shape (Batch, M, 2) containing interpolated (x, y).
    """
    # keypoints: [B, N, 3]
    # t_query: [M] (shared) or [B, M]
    
    B, N, _ = keypoints.shape
    
    # Sort by time to ensure monotonicity for searchsorted
    # keypoints: [B, N, 3]
    t = keypoints[:, :, 2]
    sorted_indices = torch.argsort(t, dim=1)
    
    # Gather all to keep correspondence
    # indices: [B, N] -> [B, N, 3]
    gather_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
    keypoints_sorted = torch.gather(keypoints, 1, gather_indices)
    
    x = keypoints_sorted[:, :, 0]
    y = keypoints_sorted[:, :, 1]
    t = keypoints_sorted[:, :, 2]
    
    if t_query.dim() == 1:
        t_query = t_query.unsqueeze(0).expand(B, -1) # [B, M]
    
    M = t_query.shape[1]
    
    # Find indices for interpolation
    # We want idx such that t[:, idx] <= t_query < t[:, idx+1]
    # Since t is sorted, we can use searchsorted
    
    # t is [B, N], t_query is [B, M]
    # torch.searchsorted expects sorted sequence
    # result is indices into t such that t[i-1] < v <= t[i] (default right=False)
    # or t[i] >= v
    
    # We need to handle the case where t_query is out of bounds
    # Clamp t_query to [t_min, t_max] of each batch item to avoid index errors,
    # but ideally the model should learn to cover the time range.
    # For now, we clamp indices.
    
    indices = torch.searchsorted(t, t_query, right=True) # [B, M]
    indices = indices - 1
    
    # Clamp indices to [0, N-2]
    indices = indices.clamp(0, N - 2)
    
    # Gather values
    # indices: [B, M]
    # We need to gather from x, y, t which are [B, N]
    
    # Helper to gather: output[b, m] = input[b, indices[b, m]]
    def gather_val(input_tensor, idx):
        return torch.gather(input_tensor, 1, idx)
    
    t0 = gather_val(t, indices)       # [B, M]
    t1 = gather_val(t, indices + 1)   # [B, M]
    x0 = gather_val(x, indices)
    x1 = gather_val(x, indices + 1)
    y0 = gather_val(y, indices)
    y1 = gather_val(y, indices + 1)
    
    # Calculate weights
    # Avoid division by zero
    dt = t1 - t0
    dt = torch.where(dt < 1e-6, torch.ones_like(dt) * 1e-6, dt)
    
    w = (t_query - t0) / dt
    w = w.clamp(0.0, 1.0) # Clamp weights for extrapolation safety
    
    # Interpolate
    x_interp = x0 + w * (x1 - x0)
    y_interp = y0 + w * (y1 - y0)
    
    return torch.stack([x_interp, y_interp], dim=-1) # [B, M, 2]


class DifferentiableSTLConverter:
    def __init__(self, rooms_dict):
        self.rooms = rooms_dict
        
    def get_room_name(self, item):
        for key in ['target', 'room', 'area', 'location', 'nearest_room']:
            if key in item and item[key] is not None:
                return item[key]
        return None

    def create_room_predicate(self, room_name):
        if room_name not in self.rooms:
            # Fallback or error
            # For robustness, maybe return a dummy predicate or error
            # Assuming room names are correct as per dataset
            return None

        x_min, x_max, y_min, y_max = self.rooms[room_name]
        
        def in_room(states):
            # states: [B, T, 2] or [T, 2]
            x, y = states[..., 0], states[..., 1]
            
            dx_min = x - x_min
            dx_max = x_max - x
            dy_min = y - y_min
            dy_max = y_max - y
            
            # Soft min for better gradients? Or hard min?
            # Hard min is fine for STL usually
            robustness = torch.min(torch.min(dx_min, dx_max), 
                                  torch.min(dy_min, dy_max))
            return robustness
        
        return Predicate(f"in_{room_name}", predicate_function=in_room)
    
    def time_to_steps(self, time_bound, dt=0.1):
        if time_bound is None:
            return None
        if isinstance(time_bound, list):
            return [int(t / dt) for t in time_bound]
        return int(time_bound / dt)
    
    def build_stl_formula(self, stl_json, dt=0.1):
        formulas = []
        if 'temporal_constraints' in stl_json:
            for constraint in stl_json['temporal_constraints']:
                try:
                    formula = self._build_constraint(constraint, dt)
                    if formula is not None:
                        formulas.append(formula)
                except Exception as e:
                    # print(f"Warning: Failed to build constraint: {e}")
                    pass
        
        if 'global_constraints' in stl_json:
            for constraint in stl_json['global_constraints']:
                try:
                    formula = self._build_global_constraint(constraint, dt)
                    if formula is not None:
                        formulas.append(formula)
                except Exception as e:
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
    
    def _build_constraint(self, constraint, dt):
        if constraint['type'] == 'sequence':
            return self._build_sequence(constraint, dt)
        elif constraint['type'] == 'choice':
            return self._build_choice(constraint, dt)
        return None
    
    def _build_sequence(self, constraint, dt):
        tasks = constraint['tasks']
        absolute_constraints = []
        
        # 1. Absolute constraints
        for task in tasks:
            room_name = self.get_room_name(task)
            if not room_name: continue
            room_pred = self.create_room_predicate(room_name)
            if room_pred is None: continue
            
            base_pred = room_pred > 0
            if 'actions' in task:
                for action in task['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time is not None and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])

            if 'time_bound' in task and task['time_bound'] is not None:
                time_bound = task['time_bound']
                if isinstance(time_bound, list):
                    interval = self.time_to_steps(time_bound, dt)
                    if isinstance(interval, list) and len(interval) == 2:
                        interval[1] += 1
                    absolute_constraints.append(Eventually(base_pred, interval=interval))
                else:
                    steps = self.time_to_steps(time_bound, dt)
                    absolute_constraints.append(Eventually(base_pred, interval=[0, steps + 1]))
            else:
                absolute_constraints.append(Eventually(base_pred))

        # 2. Ordering constraints
        ordering_formula = None
        for i in range(len(tasks) - 1, -1, -1):
            task = tasks[i]
            room_name = self.get_room_name(task)
            if not room_name: continue
            room_pred = self.create_room_predicate(room_name)
            if room_pred is None: continue
            
            base_pred = room_pred > 0
            if 'actions' in task:
                for action in task['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time is not None and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])
            
            if ordering_formula is None:
                ordering_formula = Eventually(base_pred)
            else:
                ordering_formula = Eventually(base_pred & ordering_formula)
        
        final_formula = ordering_formula
        for c in absolute_constraints:
            final_formula = final_formula & c
            
        return final_formula

    def _build_choice(self, constraint, dt):
        options = constraint['options']
        formulas = []
        for option in options:
            room_name = self.get_room_name(option)
            if not room_name: continue
            room_pred = self.create_room_predicate(room_name)
            if room_pred is None: continue
            
            base_pred = room_pred > 0
            if 'actions' in option:
                for action in option['actions']:
                    if 'wait' in action or action.get('type') == 'wait':
                        wait_time = action.get('wait', action.get('duration', 0))
                        if wait_time is not None and wait_time > 0:
                            wait_steps = self.time_to_steps(wait_time, dt)
                            base_pred = base_pred & Always(room_pred > 0, interval=[0, wait_steps])

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
        
        if len(formulas) == 0: return None
        elif len(formulas) == 1: return formulas[0]
        else:
            combined = formulas[0]
            for f in formulas[1:]:
                combined = combined | f
            return combined

    def _build_global_constraint(self, constraint, dt):
        # Placeholder for global constraints if needed
        return None

# -----------------------------------------------------------------------------
# 2. Dataset
# -----------------------------------------------------------------------------

class RobotTrajectoryDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512, max_traj_len=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_traj_len = max_traj_len
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text Processing
        instruction = item['instruction']
        output = item['output']
        
        # Tokenize instruction and output
        # We want to train on output, conditioned on instruction
        # Format: [BOS] instruction [EOS] output [EOS] ? Or just concatenation
        # Qwen usually uses ChatML or similar, but here we do simple concatenation
        
        full_text = f"{instruction}\n{output}"
        
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        # Create labels: mask out instruction part
        labels = input_ids.clone()
        
        # Find where instruction ends. This is approximate if truncation happens.
        # Better way: tokenize instruction separately to find its length
        instr_enc = self.tokenizer(
            f"{instruction}\n",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        instr_len = instr_enc['input_ids'].shape[1]
        
        if instr_len < self.max_length:
            labels[:instr_len] = -100
        else:
            # If instruction is too long, we might mask everything or handle it differently
            # Here we assume instruction fits
            pass
            
        labels[attention_mask == 0] = -100 # Mask padding
        
        # Trajectory Processing
        pwl = item['pwl'] # [[x, y, t], ...]
        # Pad or truncate pwl to max_traj_len
        pwl_tensor = torch.tensor(pwl, dtype=torch.float32)
        if pwl_tensor.shape[0] > self.max_traj_len:
            pwl_tensor = pwl_tensor[:self.max_traj_len]
        else:
            padding = torch.zeros(self.max_traj_len - pwl_tensor.shape[0], 3)
            pwl_tensor = torch.cat([pwl_tensor, padding], dim=0)
            
        # STL JSON
        stl_json_str = item.get('stl_json', '{}')
        # Remove markdown code blocks if present
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
            'stl_json': stl_json,
            'instr_len': instr_len
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    pwl = torch.stack([item['pwl'] for item in batch])
    stl_jsons = [item['stl_json'] for item in batch]
    instr_lens = torch.tensor([item['instr_len'] for item in batch], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pwl': pwl,
        'stl_jsons': stl_jsons,
        'instr_lens': instr_lens
    }

# -----------------------------------------------------------------------------
# 3. Model
# -----------------------------------------------------------------------------

class MultimodalRobotModel(nn.Module):
    def __init__(self, model_path, max_traj_len=50):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.llm.gradient_checkpointing_enable()
        
        # Use the internal model as backbone if possible, or just use the CausalLM
        # Qwen CausalLM output hidden states
        
        self.hidden_size = self.llm.config.hidden_size
        self.max_traj_len = max_traj_len
        
        # Regression Head
        # Maps hidden state to trajectory [x, y, t] * max_traj_len
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_traj_len * 3)
        )
        
        # Cast regression head to bfloat16 to match model
        self.regression_head.to(torch.bfloat16)
        
    def forward(self, input_ids, attention_mask, instr_lens=None, labels=None):
        # Get LLM outputs
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states[-1] # Last layer
        logits = outputs.logits
        
        # Regression
        # We need to pick a token to represent the instruction for regression.
        # We use the last token of the instruction.
        batch_size = input_ids.shape[0]
        reg_input = []
        for i in range(batch_size):
            idx = min(instr_lens[i] - 1, hidden_states.shape[1] - 1)
            reg_input.append(hidden_states[i, idx])
        reg_input = torch.stack(reg_input) # [B, H]
        
        traj_flat = self.regression_head(reg_input)
        traj_pred = traj_flat.view(batch_size, self.max_traj_len, 3)
        
        loss_ce = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_ce = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Return loss_ce instead of logits to save memory during gather
        return loss_ce, traj_pred

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def train():
    # Config
    MODEL_PATH = "/home/lijia/code/LLaMA-Factory/models/Qwen/Qwen3-4B"
    DATA_PATH = "/home/lijia/code/1208_CLHS/positive_robustness.json"
    BATCH_SIZE = 2 # Small batch size for debugging/memory
    LR = 1e-4
    EPOCHS = 5
    LAMBDA_CE = 1.0
    LAMBDA_MSE = 1.0
    LAMBDA_STL = 0.1
    
    # Rooms definition
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
    
    # Init
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = RobotTrajectoryDataset(DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = MultimodalRobotModel(MODEL_PATH)
    model.train()
    
    # -------------------------------------------------------------------------
    # Memory Optimization Strategy
    # -------------------------------------------------------------------------
    # 4B params full finetune requires >40GB VRAM (Weights 8GB + Grads 8GB + AdamW 32GB).
    # We must reduce memory usage.
    
    use_lora = False
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        use_lora = True
        print("Peft found. Using LoRA for memory efficient fine-tuning.")
    except ImportError:
        print("Peft not found. Freezing LLM backbone to save memory.")
        
    if use_lora:
        # Apply LoRA to LLM backbone
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=32, 
            lora_alpha=64, 
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"] # Adjust based on Qwen model
        )
        model.llm = get_peft_model(model.llm, peft_config)
        model.llm.print_trainable_parameters()
    else:
        # Freeze LLM backbone
        for param in model.llm.parameters():
            param.requires_grad = False
        print("LLM backbone frozen. Only training regression head.")

    # Ensure regression head is trainable
    for param in model.regression_head.parameters():
        param.requires_grad = True
        
    # -------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Only pass trainable parameters to optimizer to save memory
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    stl_converter = DifferentiableSTLConverter(ROOMS)

    # Fixed time grid for STL evaluation (0 to 30s)-----这不对吧
    DT = 0.5
    MAX_TIME = 60.0
    t_grid = torch.arange(0, MAX_TIME + DT, DT).to(device)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            pwl_target = batch['pwl'].to(device)
            stl_jsons = batch['stl_jsons']
            instr_lens = batch['instr_lens'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Pass labels to compute CE loss inside model (distributed)
            loss_ce_list, traj_pred = model(input_ids, attention_mask, instr_lens, labels)
            
            # 1. CE Loss
            # loss_ce_list is a tensor of shape [num_gpus] (if DataParallel) or scalar
            loss_ce = loss_ce_list.mean()
            
            # 2. MSE Loss
            # Mask out padding in target if needed, but here we assume fixed length or learn zeros
            # Better: use mask based on target length. But for now simple MSE.
            # traj_pred is float32 usually, but if model is bfloat16, it might be bfloat16.
            # Cast to float32 for loss calculation if needed, or keep as is.
            loss_mse = nn.MSELoss()(traj_pred.float(), pwl_target.float())
            
            # 3. STL Loss
            loss_stl = 0
            valid_stl_count = 0
            
            # Interpolate trajectory to dense grid for STL
            # traj_pred: [B, N, 3]
            # We need to ensure t is increasing for interpolation.
            # Force t to be increasing? Or assume model learns it.
            # To help training, we can enforce t structure or just use x,y and fixed t?
            # User said model outputs t.
            # We can sort by t just in case, or assume it's sorted.
            # Let's assume sorted for now.
            
            dense_traj = differentiable_interpolate(traj_pred, t_grid) # [B, M, 2]
            
            for i in range(len(stl_jsons)):
                stl_json = stl_jsons[i]
                formula = stl_converter.build_stl_formula(stl_json, dt=DT)
                
                if formula is not None:
                    # Evaluate robustness
                    # formula expects [M, 2] input (no batch dim)
                    trace = dense_traj[i] # [M, 2]
                    
                    # stlcgpp formula evaluation
                    # robustness is usually scalar or [M]
                    try:
                        robustness = formula.robustness(trace, scale=-1) # scale=-1 for max robustness?
                        # stlcgpp robustness: positive is satisfied.
                        # We want to maximize robustness => minimize -robustness
                        
                        # Check shape
                        if isinstance(robustness, torch.Tensor):
                            # Usually robustness is a signal over time.
                            # We want the robustness at time 0.
                            if robustness.numel() > 1:
                                score = robustness[0]
                            else:
                                score = robustness
                            loss_stl += -score
                            valid_stl_count += 1
                    except Exception as e:
                        # print(f"STL Eval Error: {e}")
                        pass
            
            if valid_stl_count > 0:
                loss_stl = loss_stl / valid_stl_count
            else:
                loss_stl = torch.tensor(0.0).to(device)
                
            # Total Loss
            loss = LAMBDA_CE * loss_ce + LAMBDA_MSE * loss_mse + LAMBDA_STL * loss_stl
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, MSE: {loss_mse.item():.4f}, STL: {loss_stl.item():.4f})")
                
        print(f"Epoch {epoch} Average Loss: {total_loss / len(dataloader)}")

    # Save model
    torch.save(model.state_dict(), "robot_model_finetuned.pth")
    print("Training complete.")

if __name__ == "__main__":
    train()
