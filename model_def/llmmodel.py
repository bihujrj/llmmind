import math
from typing import Tuple, Optional, List, Union
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from model_def.llmconfig import LlmConfig
from model_def.deepblock import DeepBlock
from model_def.pe import Rope
import torch.nn.functional as F
class LlmModel(nn.Module):
    def __init__(self,
                 config:LlmConfig):
        super().__init__()
        self.config=config
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.embed_tokens.weight=self.lm_head.weight    #输入输出权重共享，实现可逆映射
        self.ep=Rope(config.hidden_size,config.max_position_embeddings)
        self.freq_cos,self.freq_sin=self.ep.getRope()
        self.deep_layers=config.deep_layers
        self.layers=nn.ModuleList([DeepBlock(l,config) for l in range(config.deep_layers)])
        self.dropout=nn.Dropout(config.dropout)
        if torch.cuda.is_available():
            self.cuda()
    # def position_embedding(self,dim:int,
    #                        end:int=32*1024,
    #                        rope_base:float=1e6,
    #                        rope_scaling:Optional[dict]=None):
    #     #(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为1到1e6之间增函数
    #     #1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为小于1降函数，位置越靠前频率越高
    #     freqs=1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))
    #     if rope_scaling is not None:
    #         #外推,频率压缩
    #         pass
    #     t=torch.arange(end,device=freqs.device)

    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                labels: Optional[torch.Tensor] = None,
                past_kv:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache:bool=False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args
                ):
        bsize,dat_len=input_ids.shape

        embeded_dat=self.dropout(self.embed_tokens(input_ids))
        start_pose=0
        position_embedding=(self.freq_cos[start_pose:start_pose+dat_len],self.freq_sin[start_pose:start_pose+dat_len])
        present=[]

        past_kv_=past_kv or [None]*len(self.layers)
        for layer_i,(layer,past_kv_item) in enumerate(zip(self.layers,past_kv_)):
            embeded_dat,cur=layer(input=embeded_dat,past_kv=past_kv_item,use_cache=use_cache,attention_mask=attention_mask)
            present.append(cur)
        #当使用moe时，返回0张量，不影响主损失函数
        # aux_loss=sum([l.feedforward.aux_loss for l in self.layers if isinstance(l.feedforward,Moe)],embeded_dat.new_zeros(1).squeeze)
        aux_loss=0

        # return embeded_dat, present, aux_loss


        #hidden_states[:, slice_indices, :] 取出每个样本中指定位置的 hidden states
        #然后只对这些位置的 hidden states 计算 logits，从而避免了为所有 token 计算 logits 的开销。
        #在自回归生成过程中（例如循环调用模型逐个预测下一个 token），每次前向传播只需要知道当前序列最后一个 token 的 logits，用来采样下一个词。如果不加裁剪，模型会为整个输入序列的所有 token 都计算 logits，而大部分位置的 logits 是多余的，浪费了显存和计算资源。
        #训练阶段：通常需要所有 token 的 logits 来计算交叉熵损失（对比每个位置预测的词与真实词），因此应设置 logits_to_keep = 0 或负数，保留全部 logits。生成阶段：例如调用 model.generate() 时，内部会设置 logits_to_keep = 1，让模型只计算最后一个 token 的 logits，大幅提升生成速度。
        # 如果 logits_to_keep 是整数
        if isinstance(logits_to_keep, int):
            # 若大于0，构造一个从倒数第 logits_to_keep 个元素到末尾的切片
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            # 否则直接使用传入的索引（例如 slice(1,5) 或列表等）
            slice_indices = logits_to_keep
        logits = self.lm_head(embeded_dat[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 计算交叉熵损失，忽略-100的位置
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1),
                                    ignore_index=-100)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv,
            hidden_states=embeded_dat
        )


        # 附加MoE的辅助损失（可以用于日志或梯度）
        output.aux_loss = aux_loss
        return output
        # return embeded_dat,present,aux_loss



