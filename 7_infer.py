'''
.pth 是 PyTorch 的标准模型权重文件格式  ，仅保存模型的 state_dict（参数字典），不包含模型结构定义。
优点：文件较小，加载灵活，可自由选择设备（CPU/GPU）。
缺点：加载时必须重新实例化模型结构，且代码中的类定义必须与训练时完全一致。
适用场景：训练与推理分离，或需要跨平台部署。

'''

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np

# ==================== 1. 复现模型结构（必须与训练代码一致） ====================
class MultimodalRobotModel(nn.Module):
    def __init__(self, model_path, max_traj_len=50):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.hidden_size = self.llm.config.hidden_size
        self.max_traj_len = max_traj_len
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_traj_len * 3)
        )
        self.regression_head.to(torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        
        # 提取用于回归的特征：取最后一个有效 token
        # 训练时逻辑：sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        reg_input = []
        for i in range(batch_size):
            idx = sequence_lengths[i]
            reg_input.append(hidden_states[i, idx])
        reg_input = torch.stack(reg_input)
        
        traj_flat = self.regression_head(reg_input)
        traj_pred = traj_flat.view(batch_size, self.max_traj_len, 3)
        
        loss_ce = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_ce = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss_ce, traj_pred

# ==================== 2. 推理类封装 ====================
class RobotTrajectoryPredictor:
    def __init__(self, model_path, checkpoint_path, max_traj_len=50, device='auto'):
        """
        Args:
            model_path: 预训练语言模型路径
            checkpoint_path: 微调后的 .pth 权重文件路径
            max_traj_len: 最大轨迹点数
            device: 'auto'自动选择，或指定 'cuda:0'/'cpu'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = MultimodalRobotModel(model_path, max_traj_len)
        self.model.to(self.device)
        self.model.eval()
        
        self._load_checkpoint(checkpoint_path)
        self.max_traj_len = max_traj_len
        
    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        
        missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"⚠️  缺失的权重: {missing}")
        if unexpected:
            print(f"⚠️  意外的权重: {unexpected}")
        print(f"✅ 模型权重已加载: {checkpoint_path}")
    
    @torch.no_grad()
    def predict(self, instruction, max_new_tokens=512):
        """
        两步推理：
        1. 生成文本 (Reasoning + Result)
        2. 回归轨迹
        """
        # 1. 构造输入（仅指令）
        messages = [{"role": "user", "content": instruction}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 2. 生成文本
        # 使用 model.llm.generate
        generated_ids = self.model.llm.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False, # 贪婪解码，保证确定性
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 解码生成的文本（assistant的回复）
        # 需要去掉输入部分
        input_length = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 3. 回归轨迹
        # 构造全量文本：指令 + 生成的回复
        full_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": generated_text}
        ]
        full_text = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        _, traj_pred = self.model(full_inputs.input_ids, full_inputs.attention_mask)
        
        # 后处理
        traj = traj_pred[0].cpu().numpy() # [max_traj_len, 3]
        valid_mask = traj[:, 2] > 0
        trajectory = traj[valid_mask]
        
        return trajectory, generated_text

# ==================== 3. 使用示例 ====================
if __name__ == "__main__":
    MODEL_PATH = "/home/lijia/code/LLaMA-Factory/models/Qwen/Qwen3-4B"
    CHECKPOINT_PATH = "robot_model_finetuned.pth"
    
    predictor = RobotTrajectoryPredictor(
        model_path=MODEL_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        max_traj_len=50
    )
    
    test_instructions = [
        "Go to the LivingRoom and wait for 5 seconds",
        "Visit Kitchen then Bedroom in any order",
        "Move to RestRoom within 10 seconds"
    ]
    
    for instr in test_instructions:
        print(f"\n🤖 指令: {instr}")
        trajectory, response = predictor.predict(instr)
        # print(f"📝 模型回复: {response}")
        print(f"📍 生成轨迹点: {len(trajectory)} 个")
        print(f"📊 轨迹预览:\n{trajectory[:5]}")
