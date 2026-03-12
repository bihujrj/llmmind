#position embedding
import math

import torch

class Rope:
    rope_base = 10000
    attn_factor = 1.0
    input_max=2048
    def __init__(self, dim: int = 512,
                 end: int = 32 * 1024,
                 rope_scaling:bool=0):
        # 生成频率
        # freqs = 1.0 / (self.rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        # print(torch.arange(0, dim, 2)[:(dim // 2)].float() )
        # print(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)
        freqs = 1.0 / (self.rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        #NTK-aware 插值 方法，最初由 Reddit 用户 bloc97 在 2023 年提出
        # YaRN 论文（YaRN: Efficient Context Window Extension of Large Language Models）系统记录和改进
        # if rope_scaling:
        #     # 只有当序列长度超过原始最大长度时才应用缩放
        #     if end / self.input_max > 1.0:
        #         # 计算需要缩放的维度范围：根据beta_fast和beta_slow确定插值的起始和结束维度
        #         inv_dim = lambda b: (dim * math.log(self.input_max / (b * 2 * math.pi))) / (2 * math.log(self.rope_base))
        #         low = max(math.floor(inv_dim(beta_fast)), 0)
        #         high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        #         # 构建线性斜坡函数ramp，使得ramp[low]=0，ramp[high]=1，其余线性插值
        #         ramp = torch.clamp(
        #             (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
        #             0, 1
        #         )
        #         # 调整频率：freqs_new = freqs * (1 - ramp + ramp / factor)
        #         freqs = freqs * (1 - ramp + ramp / self.attn_factor)
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        # 计算 cos 和 sin，并重复拼接以匹配维度
        self.freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * self.attn_factor
        self.freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * self.attn_factor

    def getRope(self):
        return self.freqs_cos, self.freqs_sin


def test_rope():
    # 测试参数
    dim = 10
    end = 3
    rope = Rope(dim=dim, end=end)
    cos, sin = rope.getRope()

    # 1. 检查形状
    assert cos.shape == (end, dim), f"cos 形状错误: 期望 {(end, dim)}，实际 {cos.shape}"
    assert sin.shape == (end, dim), f"sin 形状错误: 期望 {(end, dim)}，实际 {sin.shape}"

    # 2. 位置 0 的编码应为全1和全0
    assert torch.allclose(cos[0], torch.ones(dim)), "位置0的 cos 应全为1"
    assert torch.allclose(sin[0], torch.zeros(dim)), "位置0的 sin 应全为0"

    # 3. 由于 cos 和 sin 是通过将频率复制拼接得到的，前半和后半应该相等
    half = dim // 2
    assert torch.allclose(cos[:, :half], cos[:, half:]), "cos 的前半与后半不相等"
    assert torch.allclose(sin[:, :half], sin[:, half:]), "sin 的前半与后半不相等"

    # 4. (可选) 检查第一个非零位置的数值是否符合预期（例如位置1）
    # 此处仅验证频率计算的基本性质：频率值应为 base^{-2i/dim}
    # 对于 dim=512，i=0 时频率为 1，i=1 时频率为 10000^{-2/512} ≈ 0.999
    # 由于计算可能因浮点误差略有差异，仅做粗略检查
    expected_freq_first = 1.0  # i=0
    expected_freq_second = 10000 ** (-2.0 / dim)  # i=1
    # 频率矩阵 freqs = t * freqs_i，t=1 时即为各频率值
    # 由于 cos 是 cat 后的结果，前半的第一个元素对应 t=1, i=0 的 cos 值
    cos_t1_i0 = cos[1, 0].item()
    cos_t1_i1 = cos[1, 1].item()
    # 理论值
    expected_cos_t1_i0 = torch.cos(torch.tensor(1.0 * expected_freq_first)).item()
    expected_cos_t1_i1 = torch.cos(torch.tensor(1.0 * expected_freq_second)).item()
    assert abs(cos_t1_i0 - expected_cos_t1_i0) < 1e-5, f"位置1、频率0的 cos 值错误"
    assert abs(cos_t1_i1 - expected_cos_t1_i1) < 1e-5, f"位置1、频率1的 cos 值错误"
    print('--sin--:')
    print(sin)
    print('--cos--:')
    print(cos)

    print("所有测试通过！")


if __name__ == "__main__":
    test_rope()


