import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_def.llmconfig import LlmConfig
from model_def.pe import Rope  # 假设你的 RoPE 实现仍在此处


class Attention(nn.Module):
    def __init__(self, config: LlmConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_head
        self.kv_heads = getattr(config, 'kv_heads', 4)  # 允许从配置读取
        self.kv_rep = config.num_attention_head // self.kv_heads  ##kv压缩倍数 for memory saving

        # 线性投影
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        # ----- RoPE 预计算缓存 -----
        max_seq_len = getattr(config, 'max_seq_len', 4096)  # 从配置获取最大长度
        rope = Rope(dim=self.head_dim, end=1000000)
        cos, sin = rope.getRope()  # 假设 getRope 返回 (cos, sin)，形状 (seq_len, head_dim)
        # 截取到最大长度并注册为 buffer（自动随模型移动设备）
        self.register_buffer("cos_cached", cos[:max_seq_len].contiguous(), persistent=False)
        self.register_buffer("sin_cached", sin[:max_seq_len].contiguous(), persistent=False)
        if torch.cuda.is_available():
            self.cuda()

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转一半维度（用于 RoPE）"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        应用旋转位置编码
        q, k: (batch, seq_len, heads, head_dim)
        cos, sin: (seq_len, head_dim)
        """
        # 扩展维度以便广播: (1, seq_len, 1, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        # https://baike.baidu.com/item/%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5/3265181    角度旋转
        q_embed = q * cos + self.rotate_half(q) * sin
        k_embed = k * cos + self.rotate_half(k) * sin
        return q_embed, k_embed

    def expand_kv(self, x: torch.Tensor, rep: int) -> torch.Tensor:
        """将 KV 头重复以匹配 query 头数"""
        if rep == 1:
            return x
        bsz, seq_len, kv_heads, head_dim = x.shape
        x = x.unsqueeze(3)  # (bsz, seq_len, kv_heads, 1, head_dim)
        x = x.expand(bsz, seq_len, kv_heads, rep, head_dim)
        return x.reshape(bsz, seq_len, kv_heads * rep, head_dim)

    def forward(
            self,
            x: torch.Tensor,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        # 1. 投影 Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 重塑为多头形式
        q = q.view(bsz, seq_len, self.config.num_attention_head, self.head_dim)
        k = k.view(bsz, seq_len, self.kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.kv_heads, self.head_dim)

        # 3. 应用 RoPE（从 buffer 中取当前序列长度的 cos/sin）
        cos = self.cos_cached[:seq_len].to(x.dtype)  # buffer 已与模型同设备
        sin = self.sin_cached[:seq_len].to(x.dtype)
        q, k = self.apply_rope(q, k, cos, sin)

        # 4. 处理 KV 缓存
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            # 更新缓存长度
            full_seq_len = k.shape[1]
        else:
            full_seq_len = seq_len

        current_kv = (k, v) if use_cache else None

        # 5. 重复 KV 头以匹配 Q 头数
        k = self.expand_kv(k, self.kv_rep)  # (bsz, full_seq_len, num_heads, head_dim)
        v = self.expand_kv(v, self.kv_rep)

        # 6. 转置为 (bsz, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 7. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 8. 因果掩码（仅对当前生成部分应用）
        if attention_mask is None:
            # 创建上三角掩码，形状 (1, 1, full_seq_len, full_seq_len)
            # Causal mask (only for current tokens; extend if past exists) 上三角置0，相当于位置掩码
            # score[:,:,:,-dat_len:]+=torch.triu(torch.full((dat_len,dat_len),float("-inf"),device=score.device),diagonal=1)
            causal_mask = torch.triu(
                torch.ones((1, 1, full_seq_len, full_seq_len), device=scores.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            scores = scores + attention_mask

        # 9. Softmax + dropout
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.attn_dropout(attn_weights)

        # 10. 加权求和
        output = torch.matmul(attn_weights, v)  # (bsz, num_heads, seq_len, head_dim)

        # 11. 恢复形状并输出投影
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(output)
        output = self.residual_dropout(output)

        return output, current_kv
