#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3.5-4B LoRA 微调脚本（原生 Transformers + PEFT）
场景：营销文案生成（小红书风格）
"""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json

# ==================== 1. 配置参数 ====================
MODEL_NAME = "Qwen/Qwen3.5-4B-Instruct"          # 模型名称
OUTPUT_DIR = "./qwen3.5-4b-lora-output"          # 输出目录
MAX_SEQ_LEN = 2048                               # 最大序列长度
BATCH_SIZE = 4                                    # 每设备 batch size
GRAD_ACCUM = 4                                    # 梯度累积步数（有效 batch = 16）
EPOCHS = 3                                        # 训练轮数
LEARNING_RATE = 2e-4                              # 学习率
LORA_R = 32                                       # LoRA 秩
LORA_ALPHA = 64                                   # LoRA 缩放参数
LORA_DROPOUT = 0.1                                # LoRA dropout
USE_4BIT = True                                   # 是否使用 4-bit 量化（节省显存）

# ==================== 2. 准备数据 ====================
# 示例数据：营销文案（可替换为你的真实数据）
train_data = [
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的小红书营销文案写手。"},
            {"role": "user", "content": "写一段蓝牙耳机的推广文案，主打降噪功能。"},
            {"role": "assistant", "content": "通勤党福音！这降噪效果绝了🚇\n\n地铁上戴上它，瞬间世界安静！主动降噪直接过滤掉90%的环境噪音，听歌完全不用开大音量。\n\n最惊喜的是通话降噪，在地铁打电话对方还以为我在图书馆📚\n\n#降噪耳机 #通勤必备 #数码好物"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的小红书营销文案写手。"},
            {"role": "user", "content": "写一段咖啡机的推广文案，主打一键操作。"},
            {"role": "assistant", "content": "懒人咖啡机天花板！每天早上多睡15分钟☕️\n\n一键出咖啡真的不是智商税！放好胶囊按一下，30秒一杯香浓拿铁就搞定了。\n\n清洗也超方便，懒人福音实锤！咖啡自由从这台小机器开始✨\n\n#咖啡机 #懒人必备 #提升幸福感好物"}
        ]
    },
    # 你可以添加更多数据...
]

# 将 messages 格式化为 ChatML 文本
def format_chat_template(example, tokenizer):
    """使用 tokenizer 的聊天模板将 messages 转为模型训练文本"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False   # 训练时不需要生成提示
    )
    return {"text": text}

# 转换为 HuggingFace Dataset
dataset = Dataset.from_list(train_data)

# ==================== 3. 加载分词器 ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 确保有 pad_token（Qwen 默认无 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 对数据集进行格式化（注意：这里提前做一次，避免训练时重复）
dataset = dataset.map(lambda x: format_chat_template(x, tokenizer))

# 查看格式化后的样例
print("格式化后的训练样例：")
print(dataset[0]["text"][:500])

# ==================== 4. 加载模型（4-bit 量化）====================
# 配置量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
) if USE_4BIT else None

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# 启用梯度检查点（节省显存）
model.gradient_checkpointing_enable()

# 准备模型进行 k-bit 训练（为 LoRA 做准备）
model = prepare_model_for_kbit_training(model)

# ==================== 5. 配置 LoRA ====================
# LoRA 配置：指定要添加适配器的模块（Qwen3.5 常用模块）
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",                 # 不训练偏置
    task_type="CAUSAL_LM",       # 因果语言模型
)

# 将 LoRA 适配器添加到模型
model = get_peft_model(model, lora_config)

# 打印可训练参数占比（LoRA 只有很少参数可训练）
model.print_trainable_parameters()

# ==================== 6. 配置训练参数 ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=10,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=not torch.cuda.is_bf16_supported(),   # 若硬件不支持 bf16 则用 fp16
    bf16=torch.cuda.is_bf16_supported(),
    remove_unused_columns=False,                # 保留 "text" 列
    report_to="none",                           # 不上传 wandb
)

# ==================== 7. 使用 SFTTrainer 进行微调 ====================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",                  # 指定文本字段
    max_seq_length=MAX_SEQ_LEN,
    packing=False,                              # 不打包（适用于短文本）
)

# 开始训练
print("开始训练...")
trainer.train()

# ==================== 8. 保存模型 ====================
# 方式一：仅保存 LoRA 适配器（轻量，约几十 MB）
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print(f"LoRA 适配器已保存至 {OUTPUT_DIR}/lora_adapter")

# 方式二（可选）：合并 LoRA 权重并保存完整模型（用于推理）
# 注意：合并需要较多显存，可以在训练完成后单独运行
# model = model.merge_and_unload()
# model.save_pretrained(f"{OUTPUT_DIR}/merged")
# tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged")

print("训练完成！")

