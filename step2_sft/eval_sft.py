import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import math
import os

def load_model(model_path: str):
    """加载微调后的模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 确保有 pad_token，通常设为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",                # 自动分配到可用设备
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

# def build_prompt(instruction: str):
# # def build_prompt(instruction: str, prompt_pre_comment: str = "",prompt_adstr: str = ""):
#     """
#     按照 Qwen 官方 chat 模板构造 prompt。
#     格式：
#         <|im_start|>user
#         {instruction}\n{input_text}<|im_end|>
#         <|im_start|>assistant
#     """
#     # input_text = "这是用户的观点:" + prompt_pre_comment + ", 结合这个观点,推荐下面的产品:" + prompt_adstr + " "
#     # user_content = f"{instruction}\n{input_text}" if input_text else instruction
#     # prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
#     # return prompt
#     input_text = "这是用户的观点:" + prompt_pre_comment + ", 结合这个观点,推荐下面的产品:" + prompt_adstr + " "
#     user_content = f"{instruction}\n{input_text}" if input_text else instruction
#     prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
#     return prompt

def build_prompt(instruction: str, input_text: str = "") -> str:
    """
    按照训练时完全相同的方式构造 prompt。
    训练代码中的格式：
        <|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n
    其中 user_content = instruction + "\n" + input_text (若 input_text 存在)，否则为 instruction
    """
    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction
    prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def generate_response(model, tokenizer, prompt: str, max_new_tokens=150, temperature=0.7, top_p=0.9):
    """生成 assistant 回复（不含 prompt 部分）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # 只取新生成的部分（去掉 prompt）
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA SFT")
    parser.add_argument('--model_path', default='../../sft_model', type=str)
    args = parser.parse_args()
    model_path = args.model_path
    print("加载模型中...")
    model, tokenizer = load_model(model_path)
    print("模型加载完成，开始测试\n" + "="*50)
    # 测试用例列表：每个元素是 (instruction, input_text)
    # test_cases = [
    #     ("介绍一下深度学习的基本概念", ""),
    #     ("用 Python 写一个快速排序函数", ""),
    #     ("解释什么是 LoRA 微调", ""),
    #     ("我最近总是失眠，有什么建议吗？", ""),
    #     # 如果有带 input 字段的样本，可以这样写：
    #     ("将以下句子翻译成英文", "今天天气很好。")
    # ]
    instruction = "你是一个营销助手，请用简洁的语言生成不超过100字的营销话术。"
    test_cases = [
        ("今天天气很好","mac"),
        ("good", "mac"),
        ("", "雀巢咖啡"),
        ("小麦长势好", "雀巢咖啡"),
        ("小麦倒扶", "雀巢咖啡"),
        ("小麦丰收", "雀巢咖啡"),
    ]
    for idx, (prompt_pre_comment, prompt_adstr) in enumerate(test_cases, 1):
        print(f"\n[测试 {idx}]")
        print(f"指令: {instruction}")
        if input_text:
            print(f"输入: {input_text}")
        input_text = "这是用户的观点:" + prompt_pre_comment + ", 结合这个观点,推荐下面的产品:" + prompt_adstr + " "
        prompt = build_prompt(instruction, input_text)
        response = generate_response(model, tokenizer, prompt)
        print(f"回答: {response}")
        print("-" * 40)




if __name__ == "__main__":
    main()