import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


"""
Qwen3.5-4B 长度外推示例

Qwen3.5-4B 原生支持 262K 上下文长度，可直接处理超长文本。
本示例包含三种长度外推方案：
1. 原生支持：直接加载模型，设置 max_new_tokens
2. RoPE 缩放外推：通过修改 rope_scaling 参数实现更长上下文
3. vLLM 高效推理：使用 vLLM 部署，支持 256K 上下文
"""

# {
#   "instruction": "请总结以下长文档的核心观点",
#   "input": "这是一篇关于人工智能发展的长篇综述。第一章介绍了AI的历史...（此处为超长文本）",
#   "output": "本文综述了人工智能的发展历程、关键技术突破和未来趋势..."
# }


"""
RoPE 缩放配置详解
# 线性缩放（推荐）
rope_scaling = {
    "type": "linear",
    "factor": 2.0  # 扩展到 524K
}

# 动态 NTK 缩放（长上下文性能更好）
rope_scaling = {
    "type": "dynamic",
    "factor": 2.0
}

# Yarn 缩放（更平滑的位置编码）
rope_scaling = {
    "type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 262144
}
"""

"""
vLLM 完整部署命令
# 单卡部署
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000
"""



# ============================================================
# 方案一：原生长上下文支持（Qwen3.5-4B 原生支持 262K）
# ============================================================
def native_long_context_example():
    """直接使用原生长上下文能力"""
    print("=" * 60)
    print("方案一：原生长上下文支持")
    print("=" * 60)

    model_name = "Qwen/Qwen3.5-4B"

    # 加载模型（使用 bfloat16 节省显存）
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 准备长文本示例（模拟长文档）
    long_text = """
    第一章：引言
    随着人工智能技术的快速发展，大语言模型在各个领域展现出强大的能力。
    Qwen3.5-4B 是阿里巴巴通义实验室于 2026 年 3 月发布的最新一代模型，
    支持高达 262K 的上下文长度，可以一次性处理整本书籍的内容。

    第二章：长上下文技术原理
    长上下文能力的实现依赖于以下几个关键技术：
    1. 分组查询注意力（GQA）：通过分组共享 KV 缓存，减少显存占用
    2. RoPE 位置编码：支持通过旋转位置编码进行长度外推
    3. 高效注意力机制：如 Flash Attention 2，加速长序列计算

    第三章：应用场景
    长上下文能力使得以下应用成为可能：
    - 整本书籍摘要生成
    - 长期对话记忆
    - 大规模代码库分析
    - 法律文档处理
    - 学术论文综述
    """ * 30  # 重复 30 次，模拟长文本

    print(f"输入文本长度: {len(long_text)} 字符")
    print(f"预估 token 数: {len(tokenizer.encode(long_text))}")

    # 构建消息
    messages = [
        {"role": "system", "content": "你是一个专业的长文本分析助手。"},
        {"role": "user", "content": f"请总结以下文档的核心内容：\n\n{long_text}"}
    ]

    # 应用聊天模板
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    print(f"实际 token 数: {inputs['input_ids'].shape[1]}")

    # 生成（注意：max_new_tokens 可以根据需要调整）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print("\n模型响应:")
    print(response)

    return model, tokenizer


# ============================================================
# 方案二：RoPE 缩放外推（用于超出原生上下文的情况）
# ============================================================
def rope_scaling_example():
    """通过 RoPE 缩放实现长度外推"""
    print("\n" + "=" * 60)
    print("方案二：RoPE 缩放外推（扩展上下文至 512K）")
    print("=" * 60)

    model_name = "Qwen/Qwen3.5-4B"

    # 配置 RoPE 缩放参数
    rope_scaling_config = {
        "type": "linear",  # 线性缩放
        "factor": 2.0  # 缩放因子，2.0 表示将原生 262K 扩展到 524K
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 使用 rope_scaling 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        rope_scaling=rope_scaling_config  # 关键：启用 RoPE 缩放
    )

    print(f"RoPE 缩放配置: {rope_scaling_config}")
    print("注意：缩放后最大上下文变为原生长度 × factor")

    # 创建超长测试文本
    very_long_text = """
    这是一段用于测试超长上下文处理的文本。Qwen3.5-4B 通过 RoPE 缩放技术，
    可以在原生 262K 的基础上进一步扩展上下文长度。线性缩放方法将位置编码
    进行均匀拉伸，使得模型能够处理更长的序列。
    """ * 50

    print(f"输入 token 数: {len(tokenizer.encode(very_long_text))}")

    messages = [{"role": "user", "content": f"请用一句话总结：{very_long_text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print("\n模型响应:")
    print(response)

    return model


# ============================================================
# 方案三：vLLM 高效部署（推荐用于生产环境）
# ============================================================
def vllm_deployment_example():
    """
    使用 vLLM 部署服务，支持 256K 上下文
    注意：此方案需要单独运行 vLLM 服务

    启动命令：
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3.5-4B \\
        --max-model-len 262144 \\
        --enable-prefix-caching \\
        --gpu-memory-utilization 0.9
    """
    print("\n" + "=" * 60)
    print("方案三：vLLM 高效部署")
    print("=" * 60)

    print("""
    vLLM 部署步骤：

    1. 安装 vLLM:
       pip install vllm

    2. 启动服务（支持 256K 上下文）:
       python -m vllm.entrypoints.openai.api_server \\
           --model Qwen/Qwen3.5-4B \\
           --max-model-len 262144 \\
           --enable-prefix-caching \\
           --gpu-memory-utilization 0.9

    3. 客户端调用示例：
    """)

    # 客户端调用代码
    client_code = """
    import openai

    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-4B",
        messages=[
            {"role": "user", "content": "请分析这段长文本..."}
        ],
        max_tokens=4096,
        temperature=0.7
    )

    print(response.choices[0].message.content)
    """
    print(client_code)

    print("\n关键配置说明:")
    print("- --max-model-len: 设置最大上下文长度（Qwen3.5 最大 262144）")
    print("- --enable-prefix-caching: 启用前缀缓存，加速重复 prompt")
    print("- --gpu-memory-utilization: 控制显存使用率")


# ============================================================
# 辅助函数：计算 token 数和显存使用
# ============================================================
def token_count_demo():
    """演示不同长度文本的 token 消耗"""
    print("\n" + "=" * 60)
    print("Token 消耗估算")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

    # 常见长度对照
    examples = {
        "一句话": "人工智能正在改变世界。",
        "一段话": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解等。",
        "一页纸（约500字）": "人工智能" * 250,
        "长文档（约10000字）": "人工智能技术发展迅速。" * 2000,
    }

    for name, text in examples.items():
        tokens = len(tokenizer.encode(text))
        print(f"{name}: {tokens} tokens")

    print("\nQwen3.5-4B 上下文容量:")
    print("- 原生支持: 262,144 tokens")
    print("- 约合: 约 20 万字中文")
    print("- 可处理: 一整本《三体》第一部")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Qwen3.5-4B 长度外推示例")
    print("注意：请确保有足够的 GPU 显存（建议 24GB+）\n")

    # 选项选择
    print("请选择运行方案:")
    print("1. 原生长上下文支持")
    print("2. RoPE 缩放外推")
    print("3. vLLM 部署示例（仅展示配置）")
    print("4. Token 消耗估算")

    choice = input("\n输入选择 (1/2/3/4): ").strip()

    if choice == "1":
        native_long_context_example()
    elif choice == "2":
        rope_scaling_example()
    elif choice == "3":
        vllm_deployment_example()
    elif choice == "4":
        token_count_demo()
    else:
        print("无效选择，运行默认示例...")
        token_count_demo()





