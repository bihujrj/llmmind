import math
from typing import Tuple, Optional, List
import torch
from torch import nn
from transformers import PretrainedConfig
from model_def.llmconfig import LlmConfig
from model_def.attention import Attention
from model_def.pe import Rope
import torch.nn.functional as F



class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.eps=eps
        # 创建一个可训练的参数，初始化为全1向量（长度=dim）
        self.weight=nn.Parameter(torch.ones(dim))  #找最优缩放因子
        pass

    def _norm(self,x):
        #等价于return x/torch.sqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        # 使用 rsqrt 更高效：x * rsqrt(mean(x^2) + eps)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)