import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False


def get_absolute_position_encoding(max_position: int, d_model: int):
    """
    生成Transformer原版绝对位置编码（正弦余弦版）
    :param max_position: 最大位置数
    :param d_model: 模型特征维度
    :return: [max_position, d_model] 位置编码矩阵
    """
    # 初始化位置编码矩阵
    pos_encoding = np.zeros((max_position, d_model))

    # 生成位置索引（0到max_position-1）
    position = np.arange(max_position)[:, np.newaxis]

    # 计算维度缩放因子（Transformer原版公式）
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    # 偶数维度用sin，奇数维度用cos
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding


# 1. 生成绝对位置编码数据
max_pos = 100  # 展示前100个位置
d_model = 128  # 特征维度（和之前RoPE保持一致）
abs_pos_emb = get_absolute_position_encoding(max_pos, d_model)

# 2. 绘制关键维度的曲线（选前4个维度，和RoPE对比）
plt.figure(figsize=(12, 6))
dim_indices = [0, 1, 2, 3]  # 选择前4个维度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, dim_idx in enumerate(dim_indices):
    plt.plot(
        np.arange(max_pos),  # x轴：位置
        abs_pos_emb[:, dim_idx],  # y轴：该维度的绝对位置编码值
        label=f'维度 {dim_idx}',
        color=colors[i],
        linewidth=2
    )

# 3. 图表美化
plt.title('Transformer 绝对位置编码（APE）曲线', fontsize=16, pad=20)
plt.xlabel('位置 (Position)', fontsize=12)
plt.ylabel('绝对位置编码值', fontsize=12)
plt.xlim(0, max_pos - 1)
plt.ylim(-1.2, 1.2)  # 匹配RoPE的y轴范围，方便对比
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

# 4. 保存图片（无数据类型错误）
plt.savefig('absolute_position_encoding.png', dpi=150, bbox_inches='tight')
plt.show()