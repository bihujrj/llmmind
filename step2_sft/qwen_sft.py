# 安装所需库（如果尚未安装）
# pip install torch transformers datasets accelerate peft bitsandbytes

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

# ==================== 配置参数 ====================
model_name = "Qwen/Qwen2.5-4B"  # 实际模型名，可根据需要调整
output_dir = "./qwen2.5-4b-sft"  # 模型保存路径
dataset_name = "yahma/alpaca-cleaned"  # 示例数据集，也可替换为自己的数据

# 量化配置（4-bit）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ==================== 加载模型和分词器 ====================
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Qwen 系列推荐设置 pad_token 为 eos_token
tokenizer.pad_token = tokenizer.eos_token
# 设置对话模板（如果需要，可以手动指定，但一般模型自带）
# tokenizer.chat_template = ...  # 通常模型已自带，无需手动设置

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 为 k-bit 训练准备模型（适用于量化后的模型）
model = prepare_model_for_kbit_training(model)

# ==================== 配置 LoRA ====================
lora_config = LoraConfig(
    r=16,               # LoRA 秩
    lora_alpha=32,      # 缩放参数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 常见模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数量

# ==================== 准备数据集 ====================
def format_instruction(example):
    """将数据集样本格式化为指令-输入-输出形式，并应用聊天模板"""
    # 构造消息列表，使用 chat_template 所需的格式
    messages = []
    if example.get("input"):
        # 带输入的指令
        messages.append({"role": "user", "content": f"{example['instruction']}\n{example['input']}"})
    else:
        messages.append({"role": "user", "content": example["instruction"]})
    messages.append({"role": "assistant", "content": example["output"]})
    # 应用 tokenizer 的聊天模板，得到文本
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

# 加载数据集（以 alpaca-cleaned 为例）
dataset = load_dataset(dataset_name, split="train")
# 切分训练/验证集
dataset = dataset.train_test_split(test_size=0.05)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 格式化数据集
train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_instruction, remove_columns=eval_dataset.column_names)

# 查看一个样本
print("训练样本示例：")
print(train_dataset[0]["text"])

# ==================== 数据整理器 ====================
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding=False)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 数据整理器（用于动态填充）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==================== 设置训练参数 ====================
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,      # 根据显存调整
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,       # 梯度累积，等效增大 batch size
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,                           # 启用混合精度
    report_to="none",                     # 禁用 wandb 等报告
)

# ==================== 初始化 Trainer ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ==================== 开始训练 ====================
trainer.train()

# ==================== 保存模型 ====================
# 保存最终的 LoRA 权重
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"模型已保存至 {output_dir}")