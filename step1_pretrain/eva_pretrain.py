import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from model_def.llmconfig import LlmConfig
from model_def.llmmodel import LlmModel
from utils.train_tools import setup_seed


def ini_model(args):
    # tokenizer=AutoTokenizer(args.token_path)
    tokenizer=AutoTokenizer.from_pretrained(args.token_path,local_files_only=True)
    # model=AutoModelForCausalLM.from_pretrained(args.pretrain_path,trust_remote_code=True)
    model = LlmModel(LlmConfig(
        hidden_size=args.hidden_size,
        num_deep_layers=args.num_deep_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    ))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.pretrain_path}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    modeldata=torch.load(ckp, map_location=args.device),
    # model.load_state_dict(modeldata, strict=True)
    if isinstance(modeldata, tuple):
        # Common pattern: (state_dict, optimizer_state, epoch, etc.)
        # Assuming state_dict is the first element
        state_dict = modeldata[0]
    elif isinstance(modeldata, dict) and 'state_dict' in modeldata:
        # Another common pattern: {'state_dict': ..., 'optimizer': ..., 'epoch': ...}
        state_dict = modeldata['state_dict']
    elif isinstance(modeldata, dict):
        # Direct state dict
        state_dict = modeldata
    else:
        model.load_state_dict(modeldata, strict=True)
        #raise TypeError(f"Unexpected model data type: {type(modeldata)}")
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(args.device),tokenizer


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
    parser.add_argument('--pretrain_path',default='../../out', type=str,
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    # parser.add_argument('--num_hidden_layers', type=int, default=6, help='隐藏层数')
    parser.add_argument('--num_deep_layers', default=6, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', type=int, default=0, choices=[0,1], help='是否使用MoE')
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true',
                        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--token_path', default='../model_def', type=str,
                        help="tokenizer ）")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    conversation=[]
    model,tokenizer=ini_model(args)
    streamer=TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True )
    for prompt in prompts:
        setup_seed(2026)
        conversation.clear()
        conversation.append({"role":"user","content":prompt})

        templates={"conversation":conversation,"tokenizer":False,"add_generation_prompt":True}
        inputs=tokenizer.bos_token+prompt
        inputs=tokenizer(inputs,return_tensors="pt",truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        generated_ids=model.generate(
            input=inputs["inputs_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=3,
            templates=0.9,
            repetition_penalty=1.0
        )
        r1=generated_ids[0]
        index=len(inputs["input_ids"][0])
        response=tokenizer.decode(r1[index:],skip_special_tokens=True )
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')



    pass

if __name__=="__main__":
    main()

