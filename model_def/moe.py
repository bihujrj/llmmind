import math
from typing import Tuple, Optional, List, Union
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from model_def.feedforward import FeedForward
from model_def.llmconfig import LlmConfig


class MoeGate(nn.Module):
    def __init__(self,config:LlmConfig):
        super().__init__()
        self.config=config
        self.top_k=config.num_experts_topk
        self.n_experts=config.n_experts

        self.alpha=config.moegate_loss_alpha
        self.seq_moe_loss=config.seq_moe_loss

        self.weight=nn.Parameter(torch.empty(self.n_experts,config.hidden_size))
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))

    def forward(self,x):
        bsize,data_len,hdim=x.shape #[4,340,64]
        x=x.view(-1,hdim) #[1360,64]

        logits=F.linear(x,self.weight,None)#[1360,7]
        scores=logits.softmax(dim=-1)

        #取最相关expert
        topk_weight,topk_idx=torch.topk(scores,k=self.top_k,dim=-1,sorted=False)

        #归一化
        denominaotr=topk_weight.sum(dim=-1,keepdim=True)+1e-20
        topk_weight=topk_weight/denominaotr #[1360,3]

        if self.training and self.alpha>0:
            scores_moe=scores
            topk_idx_moe=topk_idx.view(bsize,-1)#[4,1020]
            if self.seq_moe_loss:#按字符计算损失
                # score_seq_moe=scores_moe.view(bsize,data_len,-1)#[4,340,7]
                # ce=torch.zeros(bsize,self.n_experts,device=x.device)#[4,7]
                # #给不同expert计数
                #
                # #ce.scatter_add_(1,score_seq_moe,torch.ones(bsize,data_len*self.top_k,device=x.device)).div_(data_len*self.top_k/self.n_experts)
                # # ce.scatter_add_(1, topk_idx_moe, torch.ones_like(topk_idx_moe, dtype=torch.float)).div_(
                # #     data_len * self.top_k / self.n_experts)
                # # moe_loss=(ce*score_seq_moe(dim=-1).sum(1).mean()*self.alpha)
                # # topk indices: (bsize, data_len*top_k)
                # topk_idx = topk_idx.view(bsize, -1)

                # scores: (bsize, data_len, n_experts)
                scores = scores_moe.view(bsize, data_len, -1)#[4,340,7]
                # topk indices: (bsize, data_len*top_k)
                topk_idx = topk_idx.view(bsize, -1) #[4,1020] 1020个变量在0-6之间

                # Count how many times each expert is selected (per batch)
                counts = torch.zeros(bsize, self.n_experts, device=x.device)#[4,7]
                counts.scatter_add_(1, topk_idx, torch.ones_like(topk_idx, dtype=torch.float)) #[4,7]

                # Normalize counts: sum over experts = n_experts per batch
                total_tokens = data_len * self.top_k #430*3=1020
                fi = counts * (self.n_experts / total_tokens)  # shape (bsize, n_experts)  [4,7]    选中长度/数据长度

                # Average score per expert per batch
                pi = scores.mean(dim=1)  # (bsize, n_experts)[4,7]

                # Load‑balancing loss: average over batches
                moe_loss = (pi * fi).sum(dim=1).mean() * self.alpha#pi点积fi变成标量
            else:
                # mask_ce=F.one_hot(topk_idx_moe.view(-1),num_classes=self.n_experts)
                # ce=mask_ce.float().mean(0)
                # pi=scores_moe.mean(0)
                # fi=ce*self.n_experts
                # moe_loss=(pi*fi).sum()*self.alpha
                # original non‑seq branch
                mask_ce = F.one_hot(topk_idx_moe.view(-1), num_classes=self.n_experts)
                ce = mask_ce.float().mean(0)  # fraction per expert
                pi = scores_moe.mean(0)  # average score per expert
                fi = ce * self.n_experts  # scale to sum = n_experts
                moe_loss = (pi * fi).sum() * self.alpha
        else:
            moe_loss=scores.new_zeros(1).squeeze()
        return topk_idx,topk_weight,moe_loss


class MoeFeedForward(nn.Module):
    def __init__(self,config:LlmConfig):
        super().__init__()
        self.config=config
        self.experts=nn.ModuleList([
            FeedForward(config) for _ in range(config.n_experts)
        ])
        self.gate=MoeGate(config)
        if config.n_share_experts>0:
            self.share_experts=nn.ModuleList([
                FeedForward(config) for _ in range(config.n_experts)
            ])
        pass

    def forward(self,x):
        x_bak=x
        org_shape=x.shape
        topk_idx,topk_weight,moe_loss=self.gate(x)
        #输入张量 x 通常的形状为 (batch_size, seq_len, hidden_dim)。该行代码会将其展平为 (batch_size * seq_len, hidden_dim)，
        #使得每个 token 独立成为一个样本，方便后续的线性层、门控网络或专家网络进行逐 token 处理。
        #x = torch.randn(2, 3, 512)  # (batch=2, seq=3, hidden=512)
        #x = x.view(-1, x.shape[-1]) # 形状变为 (6, 512)
        x=x.view(-1,x.shape[-1])
        flat_topk_idx=topk_idx.view(-1)
        if self.training:
            # 训练时：由于每个token可能被多个专家处理，需要复制输入
            # 将x重复top_k次，以便每个专家副本独立计算
            #x = torch.randn(4, 512)          # 4 个 token, hidden=512
            #k = 2
            #x = x.repeat_interleave(k, dim=0) # 形状 [8, 512]
            # x_repeated=x.repeat_interleave(self.config.num_experts_topk,dim=0)
            # y=torch.empty_like(x,dtype=x.dtype)
            N = x.shape[0]  # total tokens = batch_size * seq_len
            top_k = self.config.num_experts_topk
            y = torch.zeros(N, top_k, x.shape[-1], dtype=x.dtype, device=x.device)
            x_repeated = x.repeat_interleave(top_k, dim=0)
            flat_topk_idx = topk_idx.view(-1)
            # 对每个专家，处理分配给它的token
            for i,expert in enumerate(self.experts):
                # 找到当前专家处理的token索引
                mask = (flat_topk_idx == i)
                # if mask.any():
                #     expert_out = expert(x_repeated[mask])
                #     y[mask] = expert_out.to(y.dtype)
                if mask.any():
                    pos = mask.nonzero(as_tuple=True)[0]  # indices in the flattened list
                    token_idx = pos // self.config.num_experts_topk  # which original token
                    slot_idx = pos % self.config.num_experts_topk   # which top‑k slot
                    expert_out = expert(x_repeated[mask])
                    # Assign the output to the correct (token, slot) position
                    y[token_idx, slot_idx] = expert_out
                # 如果没有token分配给该专家，y中对应位置保持不变（但后续会乘以权重）
                # 注意：对于没有token的专家，y中对应位置可能还是未初始化的，但乘以权重后可能被忽略？
                # 这里为了避免未初始化问题，可以加一个小的常数项，但代码中加了0*sum(p)来确保梯度计算？
                # 实际上代码中有一行：else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
                # 但这行在else里？我们看到的代码是在if内部和else都有？可能原始代码有误。
                # 根据提供的代码，它是：
                # if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                # else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
                # 这里else条件永远不满足，因为expert_out.shape[0]>0才进入if，else不会执行。
                # 实际上可能想处理expert_out为空的情况？但这里逻辑有点混乱。
                # 按照标准实现，应该只处理非空的情况，其他位置保持0，因为后续要加和。
                # 这里为了安全，我们注释说明：当expert_out为空时，y中对应位置保持原样（可能是未初始化），
                # 但后续会被乘以权重，而权重可能为0？但topk_weight中对应位置是存在的，所以这里需要初始化y为0。
                # 代码中y是用torch.empty_like创建的，所以未初始化的部分可能是任意值，这可能导致问题。
                # 可能需要在循环前将y初始化为0，但代码中没有。这里按照原始代码不做修改，但指出潜在风险。
            # y = y.view(*topk_weight.shape, -1)  # [bsz*seq_len, top_k, hidden]
            # y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)  # 加权求和
            # y = y.view(*org_shape)
            # Apply weights and sum over slots
            y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)  # -> (N, hidden)
            y = y.view(*org_shape)  # -> (batch, seq, hidden)
        else:
            # 推理时，使用优化的moe_infer方法避免重复计算
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*org_shape)
        # 添加共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(x_bak)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理时的高效MoE计算：根据专家索引对token进行分组，然后每个专家处理一组token。
        避免了训练时复制token的开销。
        """
        expert_cache = torch.zeros_like(x)  # 初始化输出缓存
        # 按专家索引排序，以便连续处理同一专家的token
        idxs = flat_expert_indices.argsort()
        # 统计每个专家处理的token数量，并计算累积和，用于划分区间
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 原始token索引（去重后的token索引，因为每个token可能被多个专家处理，但这里flat_expert_indices已经包含了所有专家分配）
        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 该专家没有token
            expert = self.experts[i]
            # 取出分配给当前专家的token
            exp_token_idx = token_idxs[start_idx:end_idx]  # 这些是原始x中的索引（去重后？注意：每个token可能出现多次，但这里token_idxs是去重后的？）
            # 实际上，由于每个token可能被多个专家选择，token_idxs中同一个token可能出现在多个专家的区间内，
            # 但这里通过idxs排序，flat_expert_indices对应每个专家分配，token_idxs是原始token在x中的位置，
            # 所以exp_token_idx可能包含重复的token索引（如果同一个token被多个专家选择，它会在不同专家的区间中出现）。
            expert_tokens = x[exp_token_idx]  # 取出这些token的向量
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 乘以对应权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加到缓存中（注意：同一个token可能被多个专家累加）
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


