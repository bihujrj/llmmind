import argparse
import json
import math
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import gc

# 内存优化
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ----------------------------- 【已修复设备问题】LoRA 核心 -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 关键修复：创建时直接对齐设备 + 精度
        dtype = original_linear.weight.dtype
        device = original_linear.weight.device

        self.lora_A = nn.Linear(original_linear.in_features, r, bias=False, dtype=dtype, device=device)
        self.lora_B = nn.Linear(r, original_linear.out_features, bias=False, dtype=dtype, device=device)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.original_linear(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return out + lora_out

def inject_lora(model, r=8, alpha=16, dropout=0.0):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    n_,model_=list(model.named_modules())
    print(n_)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            parent_path = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1] #叶子节点
            parent = model.get_submodule(parent_path)
            lora_layer = LoRALinear(module, r, alpha, dropout)
            setattr(parent, child_name, lora_layer)  #将原来注意力替换成lora计算
    return model

# def merge_lora_weights(model):
#     for name, module in list(model.named_modules()):
#         if isinstance(module, LoRALinear):
#             parent_path = ".".join(name.split(".")[:-1])
#             child_name = name.split(".")[-1]
#             parent = model.get_submodule(parent_path)
#             setattr(parent, child_name, module.original_linear)
#     return model

def merge_lora_weights(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            # 1. 数值合并：将 LoRA 增量加到原始权重上
            with torch.no_grad():
                # 计算增量: delta_W = (lora_B.weight @ lora_A.weight) * scaling
                delta_w = (module.lora_B.weight @ module.lora_A.weight) * module.scaling
                module.original_linear.weight.add_(delta_w)
                # 如果有 bias 且 LoRA 也修改了 bias（本例中没有，但安全起见）
                # 如果 LoRA 层有 bias 处理，这里也需要合并

            # 2. 结构替换：将 LoRALinear 替换为已合并权重的 original_linear
            parent_path = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]    #叶子节点
            parent = model.get_submodule(parent_path)
            setattr(parent, child_name, module.original_linear)    #将注意力指向修改后的lora叠加的结果
    return model

# ----------------------------- 数据集 -----------------------------
class Qwen35Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if item.get("input", ""):
            user_content = f"{item['instruction']}\n{item['input']}"
        else:
            user_content = item['instruction']

        prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt + item['output'] + "<|im_end|>"

        encodings = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
        prompt_len = len(prompt_enc["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ----------------------------- 训练主函数 -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA SFT")
    parser.add_argument('--model_name', default='Qwen/Qwen3.5-4B', type=str)
    parser.add_argument('--output_path', default='./sft_model', type=str)
    parser.add_argument('--train_data', default='./sft.json', type=str)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_key', type=str)
    parser.add_argument('--LORA_R', default=8, type=int)
    parser.add_argument('--LORA_ALPHA', default=16, type=int)
    parser.add_argument('--LORA_DROPOUT', default=0.1, type=float)
    args = parser.parse_args()

    # 超低内存配置
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.03
    MAX_SEQ_LEN = 1024

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（不使用量化，避免设备问题）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=True,
    )

    # 冻结 + 注入 LoRA
    for param in model.parameters():
        param.requires_grad = False
    model = inject_lora(model, args.LORA_R, args.LORA_ALPHA, args.LORA_DROPOUT)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 数据集
    if not os.path.exists(args.train_data):
        print(f"数据不存在: {args.train_data}")
        return
    dataset = Qwen35Dataset(args.train_data, tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # 优化器
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE)
    total_steps = len(dataloader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    # 训练
    model.train()
    step_idx = 0
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_idx += 1
            pbar.set_postfix(loss=f"{outputs.loss.item():.3f}")

    # 保存
    model = merge_lora_weights(model)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("✅ 训练完成！")

if __name__ == "__main__":
    main()