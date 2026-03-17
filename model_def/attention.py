import math
from typing import Tuple, Optional
import torch
from torch import nn
from transformers import PretrainedConfig
from model_def.llmconfig import LlmConfig
from model_def.pe import Rope
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self,args:LlmConfig):
        super().__init__()
        self.head_dim=args.hidden_size//args.num_attention_head
        self.kv_head=4
        self.kv_rep=args.num_attention_head//self.kv_head   ##kv压缩倍数 for memory saving
        self.q_proj=nn.Linear(args.hidden_size,args.hidden_size)#输出要作为下一级输入
        self.k_proj=nn.Linear(args.hidden_size,self.kv_head*self.head_dim)
        self.v_proj=nn.Linear(args.hidden_size,self.kv_head*self.head_dim)
        self.o_proj=nn.Linear(args.hidden_size,args.hidden_size)
        self.attn_dropout=nn.Dropout(args.attn_dropout)
        self.residual_dropout=nn.Dropout(args.residual_dropout)
        self.num_attention_head=args.num_attention_head

    def pos_emb(self,q,k,
                cos,sin,
                unsqueeze_dim=1):
        def rotate(x):
            return torch.cat(-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2],dim=-1)
        q_embed=(q*cos.unsequenze(unsqueeze_dim+(rotate(q)*sin.unsquenze(unsqueeze_dim))))
        k_embed=(k*cos.unsequenze(unsqueeze_dim+(rotate(k)*sin.unsquenze(unsqueeze_dim))))

        return q_embed,k_embed


    def expand_kv(self,x:torch.Tensor,rep:int):
        bsize,dat_len,kv_head,head_dim=x.shape
        if rep==1:
            return x
        else:
            return x[:,:,None,:].expand(bsize,dat_len,kv_head,self.kv_rep,head_dim).reshape(bsize,dat_len,kv_head*self.kv_rep,head_dim)

    def forward(self,
                x:torch.Tensor,
                kv_cache:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
                use_cache:bool=0,
                attention_mask:Optional[torch.Tensor]=None
                ):
        bsize,dat_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        xq=xq.view(bsize,dat_len,self.num_attention_head,self.head_dim)
        xk=xk.view(bsize,dat_len,self.kv_head,self.head_dim)
        xv=xv.view(bsize,dat_len,self.kv_head,self.head_dim)

        rope = Rope(dim=512, end=1000000)
        #some kind of polar coordinates
        cos, sin = rope.getRope()
        xq,xk=self.pos_emb(xq,xk,cos,sin)

        # if kv_cache not None:
        #
        past_kv=(xk,xv) if use_cache else None

        xq,xk,xv=(
            xq.transpose(1,2),
            self.expand_kv(xk,self.kv_rep).transpose(1,2),
            self.expand_kv(xv, self.kv_rep).transpose(1, 2)
        )

        score=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
        score[:,:,:,-dat_len:]+=torch.triu(torch.full((dat_len,dat_len),float("-inf"),device=score.device),diagonal=1)

        score=F.softmax(score.float(),dim=-1).type_as(xq)
        score=self.attn_dropout(score)
        output=score@xv

        output=output.transpose(1,2).reshape(bsize,dat_len,-1)
        output=self.residual_dropout(self.o_proj(output))
        return output,past_kv



