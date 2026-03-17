import math
from typing import Tuple, Optional
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from model_def.llmconfig import LlmConfig
from model_def.pe import Rope
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self,config:LlmConfig):
        super().__init__()
        if config.feedforward_dim is None:
            #feedforward_dim取整到64倍数
            feedforward_size=int(config.hidden_size*8/3)
            config.feedforward_dim=64*((feedforward_size+64-1)//64)
        self.gate_proj=nn.Linear(config.hidden_size,config.feedforward_dim,bias=False)
        #激活函数
        self.act_fn=ACT2FN[config.feedforward_act]
        self.up_proj=nn.Linear(config.hidden_size,config.feedforward_dim,bias=False)
        self.down_proj=nn.Linear(config.feedforward_dim,config.hidden_size,bias=False)
        self.dropout=nn.Dropout(config.dropout)

    def forward(self,x):
        # return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))
        # SwiGLU: (silu(gate(x)) * up(x)) projected down
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x
