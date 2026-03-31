import argparse

import json
import math
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
# from transformers import get_linear_schedule_with_warmup
# from modelscope import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
#
# [
#   {
#     "instruction": "将以下句子翻译成英文：",
#     "input": "你好，世界！",
#     "output": "Hello, world!"
#   },
#   {
#     "instruction": "请解释什么是机器学习？",
#     "input": "",
#     "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。"
#   },
#   {
#     "instruction": "写一首关于春天的五言绝句。",
#     "input": "",
#     "output": "春风拂面来，花落知多少。万物皆苏醒，人间处处好。"
#   }
# ]

# ----------------------------- 配置 -----------------------------





# ----------------------------- 1. 手动实现 LoRA 层 -----------------------------
class LoRALinear(nn.Module):
    """手动实现的 LoRA 线性层"""

    def __init__(self, original_linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_out = self.original_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B
        lora_out = lora_out * self.scaling
        return original_out + lora_out


def inject_lora(model, r, alpha, dropout):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(model, name, LoRALinear(module, r, alpha, dropout))
        else:
            inject_lora(module, target_modules, r, alpha, dropout)
    return model


def merge_lora_weights(model):
    """将 LoRA 权重合并回原始线性层，并将模型转换为普通 Linear 结构"""
    for name, module in model.named_children():
        if isinstance(module, LoRALinear):
            # 计算合并后的权重: W + (A @ B) * scaling
            merged_weight = module.original_linear.weight.data + (
                        module.lora_A.data @ module.lora_B.data) * module.scaling
            # 创建新的普通线性层
            new_linear = nn.Linear(
                module.original_linear.in_features,
                module.original_linear.out_features,
                bias=module.original_linear.bias is not None,
                dtype=merged_weight.dtype
            )
            new_linear.weight.data = merged_weight
            if module.original_linear.bias is not None:
                new_linear.bias.data = module.original_linear.bias.data
            # 替换
            setattr(model, name, new_linear)
        else:
            merge_lora_weights(module)
    return model



# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# )




# ----------------------------- 3. 数据集 -----------------------------
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
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
        prompt_len = len(prompt_enc["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }







def main():
    parser = argparse.ArgumentParser(description="llmmind_sft")
    parser.add_argument('--model_name', default='Qwen/Qwen3.5-4B"', type=str,help="模型名")
    parser.add_argument('--output_path', default='../../llm_sft', type=str,help="输出目录")
    parser.add_argument('--train_data', default='./sft.json', type=str,help="训练文件")
    # LoRA 超参数
    parser.add_argument('--LORA_R', default=16, type=int,help="LoRA 超参数 LORA_R")
    parser.add_argument('--LORA_ALPHA', default=32, type=int,help="LoRA 超参数 LORA_ALPHA")
    parser.add_argument('--LORA_DROPOUT', default=0.1, type=float,help="LoRA 超参数 LORA_DROPOUT")


    args = parser.parse_args()

    # LoRA 超参数
    LORA_R = args.LORA_R
    LORA_ALPHA = args.LORA_ALPHA
    LORA_DROPOUT = args.LORA_DROPOUT

    # 训练配置
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.03
    MAX_SEQ_LEN = 2048
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------- 2. 加载模型和分词器 -----------------------------
    print("Loading model and tokenizer...")
    MODEL_NAME=args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 注入 LoRA
    model = inject_lora(model, LORA_R, LORA_ALPHA, LORA_DROPOUT)

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    DATA_PATH=args.train_data
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ 数据文件 {DATA_PATH} 不存在，请先创建数据文件")
        print("示例数据格式：")
        print(json.dumps([
            {"instruction": "将以下句子翻译成英文：", "input": "你好，世界！", "output": "Hello, world!"},
            {"instruction": "请解释什么是机器学习？", "input": "", "output": "机器学习是人工智能的一个分支..."}
        ], ensure_ascii=False, indent=2))
        exit(1)

    dataset = Qwen35Dataset(DATA_PATH, tokenizer, MAX_SEQ_LEN)
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
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress_bar.set_postfix({
                "loss": loss.item() * GRAD_ACCUM_STEPS,
                "lr": scheduler.get_last_lr()[0]
            })

        print(f"Epoch {epoch + 1} avg loss: {epoch_loss / len(dataloader):.4f}")

    # ----------------------------- 6. 保存 LoRA 权重 -----------------------------
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_state_dict[name] = param.data

    OUTPUT_DIR=args.output_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(lora_state_dict, os.path.join(OUTPUT_DIR, "lora_weights.pt"))
    # tokenizer.save_pretrained(OUTPUT_DIR)

    # ----------------------------- 7. 合并 LoRA 并保存完整模型 -----------------------------
    print("Merging LoRA weights into base model...")
    model = merge_lora_weights(model)

    print("Saving full model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 将模型转换为 float32 以兼容大多数量化工具
    model = model.to(torch.float32)
    model.save_pretrained(OUTPUT_DIR)

    # 保存分词器
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Full model saved to {OUTPUT_DIR}")
    print(f"Model files: {os.listdir(OUTPUT_DIR)}")



if __name__ == "__main__":
    main()
