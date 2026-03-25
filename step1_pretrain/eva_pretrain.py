import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from model_def.llmconfig import LlmConfig
from model_def.llmmodel import LlmModel, LlmForCausalLM
from utils.train_tools import setup_seed


# def ini_model(args):
#     # tokenizer=AutoTokenizer(args.token_path)
#     tokenizer=AutoTokenizer.from_pretrained(args.token_path,local_files_only=True)
#     # model=AutoModelForCausalLM.from_pretrained(args.pretrain_path,trust_remote_code=True)
#     model = LlmForCausalLM(LlmConfig(
#         hidden_size=args.hidden_size,
#         num_deep_layers=args.num_deep_layers,
#         use_moe=bool(args.use_moe),
#         inference_rope_scaling=args.inference_rope_scaling
#     ))
#     moe_suffix = '_moe' if args.use_moe else ''
#     ckp = f'./{args.pretrain_path}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
#     modeldata=torch.load(ckp, map_location=args.device),
#     #model.load_state_dict(modeldata, strict=True)
#     if isinstance(modeldata, tuple):
#         # Common pattern: (state_dict, optimizer_state, epoch, etc.)
#         # Assuming state_dict is the first element
#         state_dict = modeldata[0]
#     # elif isinstance(modeldata, dict) and 'state_dict' in modeldata:
#     #     # Another common pattern: {'state_dict': ..., 'optimizer': ..., 'epoch': ...}
#     #     state_dict = modeldata['state_dict']
#     # elif isinstance(modeldata, dict):
#     #     # Direct state dict
#     #     state_dict = modeldata
#     # else:
#     #     model.load_state_dict(modeldata, strict=True)
#     #     #raise TypeError(f"Unexpected model data type: {type(modeldata)}")
#     # Handle tuple/dict
#     if isinstance(modeldata, tuple):
#         state_dict = modeldata[0]
#     elif isinstance(modeldata, dict):
#         state_dict = modeldata.get('model_state_dict', modeldata)
#     else:
#         state_dict = modeldata
#
#     # Add "model." prefix to all keys
#     fixed_state_dict = {f'model.{k}': v for k, v in state_dict.items()}
#     model.load_state_dict(fixed_state_dict, strict=True)
#     return model.eval().to(args.device),tokenizer
def ini_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.token_path, local_files_only=True)

    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlmForCausalLM(LlmConfig(
        hidden_size=args.hidden_size,
        num_deep_layers=args.num_deep_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    ))

    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.pretrain_path}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

    # Load checkpoint
    modeldata = torch.load(ckp, map_location=args.device)

    # Handle different save formats
    if isinstance(modeldata, tuple):
        state_dict = modeldata[0]
    elif isinstance(modeldata, dict):
        state_dict = modeldata.get('model_state_dict', modeldata)
    else:
        state_dict = modeldata

    # Add "model." prefix to all keys
    fixed_state_dict = {f'model.{k}': v for k, v in state_dict.items()}

    # Load with strict=False to see any issues
    missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")

    return model.eval().to(args.device), tokenizer

def main():
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    parser = argparse.ArgumentParser(description="Llmind")
    parser.add_argument('--pretrain_path', default='../../out', type=str,
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_deep_layers', default=6, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', type=int, default=0, choices=[0,1], help='是否使用MoE')
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true',
                        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--token_path', default='../model_def', type=str,
                        help="tokenizer ）")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--show_speed', default=True, action='store_true', help="显示生成速度")
    parser.add_argument('--use_moe', type=int, default=0, choices=[0,1], help='是否使用MoE')
    args = parser.parse_args()

    model, tokenizer = ini_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    for prompt in prompts:
        setup_seed(2026)

        # Build conversation
        conversation = [{"role": "user", "content": prompt}]

        # Apply chat template to get formatted text
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        formatted_text = tokenizer.apply_chat_template(**templates)

        # Tokenize the formatted text
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()

        # Generate response
        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,  # Fixed: changed from top_p=3 to 0.9
            temperature=0.9,  # Fixed: changed from templates=0.9 to temperature=0.9
            repetition_penalty=1.0,
             use_cache=False
        )

        # Decode the generated part
        input_length = len(inputs.input_ids[0])
        response = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)

        # Calculate speed
        gen_tokens = len(generated_ids[0]) - input_length
        print(prompt+":")
        print(response)
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n')

if __name__=="__main__":
    main()

