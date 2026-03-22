import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from utils.train_tools import setup_seed


def ini_model(args):
    tokenizer=AutoTokenizer(args.token_path)
    model=AutoModelForCausalLM.from_pretrained(args.pretrain_path,trust_remote_code=True)
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
    parser.add_argument('--pretrain_path',default='../out', type=str,
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--token_path', default='model', type=str,
                        help="tokenizer ）")
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
        inputs=tokenizer.bos_toekn+prompt
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
        response=tokenizer.decode(generated_ids[0][index:],skip_special_tokens=True )
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')



    pass

if __name__=="__main__":
    main()

