from transformers import PretrainedConfig
class LlmConfig(PretrainedConfig):
    def __init__(self,
                 hidden_size=512,
                 num_attention_head:int=8,
                 attn_dropout:float=0.0,
                 residual_dropout:float=0.0,
                 ):

        self.hidden_size=hidden_size
        self.num_attention_head=num_attention_head


