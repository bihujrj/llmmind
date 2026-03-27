
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# ----------------------------- 配置 -----------------------------
MODEL_NAME = "Qwen/Qwen3.5-4B"          # 模型名称或本地路径
OUTPUT_DIR = "../qwen_lora_sft"          # 输出目录
DATA_PATH = "../yxxb_train_data.json"           # 训练数据路径（JSON格式）
LORA_R = 8                              # LoRA 秩
LORA_ALPHA = 16                         # 缩放系数
LORA_DROPOUT = 0.1                      # Dropout 概率
TARGET_MODULES = ["q_proj", "v_proj"]   # 注入 LoRA 的目标模块（Qwen2.5 的注意力投影层）
BATCH_SIZE = 1                          # 根据显存调整
GRAD_ACCUM_STEPS = 4                    # 梯度累积步数
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
MAX_SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------- 1. 手动实现 LoRA 层 -----------------------------
class LoRALinear(nn.Module):
    """手动实现的 LoRA 线性层，包装原始线性层并添加低秩分解"""
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear  # 原始权重，冻结
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 初始化 LoRA 矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r

        # 初始化 A 用 kaiming 均匀分布，B 用零（使初始时 LoRA 分支贡献为零）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原始线性层参数
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 原始输出
        original_out = self.original_linear(x)
        # LoRA 分支：x * A * B
        lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B
        lora_out = lora_out * self.scaling
        return original_out + lora_out

def inject_lora(model, target_modules, r, alpha, dropout):
    """递归替换模型中的目标线性层为 LoRALinear"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # 替换为 LoRALinear
            setattr(model, name, LoRALinear(module, r, alpha, dropout))
        else:
            inject_lora(module, target_modules, r, alpha, dropout)
    return model

# ----------------------------- 2. 加载模型和分词器 -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 设置 pad_token（若未设置）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型（使用 bfloat16 节省显存，并自动分配设备）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",          # 自动分布到可用设备
    trust_remote_code=True
)

# 冻结所有参数，准备注入 LoRA
for param in model.parameters():
    param.requires_grad = False

# 注入 LoRA 到目标模块
model = inject_lora(model, TARGET_MODULES, LORA_R, LORA_ALPHA, LORA_DROPOUT)

# 打印可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")

# ----------------------------- 3. 准备数据集 -----------------------------
class InstructionDataset(Dataset):
    """简单的指令微调数据集"""
    def __init__(self, data_path, tokenizer, max_length):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        # 构建 prompt（Qwen 推荐使用 chat 模板，这里简化为直接拼接）
        # 实际生产环境建议使用 tokenizer.apply_chat_template
        if input_text:
            prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt + output + "<|im_end|>"

        # 编码
        encodings = tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # 标签：将 prompt 部分的 token 设置为 -100（忽略 loss）
        prompt_enc = tokenizer(prompt, truncation=True, max_length=self.max_length)
        prompt_len = len(prompt_enc["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 检查数据文件是否存在
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"数据文件 {DATA_PATH} 未找到，请先准备数据。")

dataset = InstructionDataset(DATA_PATH, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------- 4. 优化器和调度器 -----------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

total_steps = len(dataloader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ----------------------------- 5. 训练循环 -----------------------------
model.train()
global_step = 0

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for step, batch in enumerate(progress_bar):
        # 将数据移动到模型所在设备（由于 device_map="auto"，输入放在主设备即可）
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        epoch_loss += loss.item() * GRAD_ACCUM_STEPS

        # 梯度累积
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            # 梯度裁剪（只对可训练参数）
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # 更新进度条
        progress_bar.set_postfix({
            "loss": loss.item() * GRAD_ACCUM_STEPS,
            "lr": scheduler.get_last_lr()[0]
        })

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

# ----------------------------- 6. 保存 LoRA 权重 -----------------------------
# 只保存 LoRA 参数（即 lora_A 和 lora_B）
lora_state_dict = {}
for name, param in model.named_parameters():
    if "lora_" in name and param.requires_grad:
        lora_state_dict[name] = param.data

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(lora_state_dict, os.path.join(OUTPUT_DIR, "lora_weights.pt"))
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA weights saved to {OUTPUT_DIR}")

# 可选：保存完整模型（注意此操作会保存所有参数，包括原始模型权重）
# model.save_pretrained(OUTPUT_DIR)