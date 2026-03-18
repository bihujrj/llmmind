import math
from typing import Tuple, Optional, List
import torch
from torch import nn
from transformers import PretrainedConfig
from model_def.llmconfig import LlmConfig
from model_def.attention import Attention
from model_def.feedforward import FeedForward
from model_def.pe import Rope
import torch.nn.functional as F

from model_def.rmsnorm import RMSNorm


class DeepBlock(nn.Module):
    def __init__(self,layer_id:int,config:LlmConfig):
        super().__init__()
        self.attn=Attention(config)
        self.num_attention_head=config.num_attention_head
        self.hidden_size=config.hidden_size
        self.head_dim=config.hidden_size//config.num_attention_head
        self.layer_id=layer_id
        self.input_norm=RMSNorm(config.hidden_size,eps=config.norm_eps)
        self.post_norm=RMSNorm(config.hidden_size,eps=config.norm_eps)
        self.feedforward=FeedForward(config)
        if torch.cuda.is_available():
            self.cuda()
    def forward(self,
                input,
                position_embedding,
                past_kv,
                use_cache=None,
                attention_mask=None):
        residual=input
        attn_output,cur_kv=self.attn(self.input_norm(input),position_embedding,past_kv,attention_mask)
        attn_output+=residual
        fd_output=attn_output+self.feedforward(self.post_norm(attn_output))
        return fd_output,cur_kv





