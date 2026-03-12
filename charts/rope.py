import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

# Llama 原生 RoPE 核心参数
def get_rope_freqs(dim: int, max_position: int = 2048, theta: float = 10000.0):
    """计算Llama RoPE的频率矩阵"""
    # 生成频率：theta^(-2(i-1)/dim)，i从1到dim/2
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    # 生成位置索引：0到max_position-1
    positions = np.arange(max_position)
    # 外积：位置 × 频率 → [max_position, dim//2]
    freqs = np.outer(positions, freqs)
    # 生成sin/cos矩阵 → [max_position, dim]
    emb = np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1)
    return emb

# 1. 生成RoPE编码数据
dim = 128  # 特征维度（取128方便可视化）
max_pos = 100  # 最大位置（取前100个位置）
rope_emb = get_rope_freqs(dim, max_pos)

# 2. 绘制关键维度的曲线（选前4个维度展示）
plt.figure(figsize=(12, 6))
dim_indices = [0, 1, 2, 3]  # 选择前4个维度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, dim_idx in enumerate(dim_indices):
    plt.plot(
        np.arange(max_pos),  # x轴：位置
        rope_emb[:, dim_idx],  # y轴：该维度的RoPE值
        label=f'维度 {dim_idx}',
        color=colors[i],
        linewidth=2
    )

# 3. 图表美化
plt.title('Llama RoPE（旋转位置编码）曲线', fontsize=16, pad=20)
plt.xlabel('位置 (Position)', fontsize=12)
plt.ylabel('RoPE 编码值', fontsize=12)
plt.xlim(0, max_pos-1)
plt.ylim(-1.2, 1.2)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

# 4. 保存图片（解决array类型错误）
plt.savefig('llama_rope_position_encoding.png', dpi=150, bbox_inches='tight')
plt.show()