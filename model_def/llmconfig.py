from transformers import PretrainedConfig
class LlmConfig(PretrainedConfig):
    model_type = "llm"  # Set a model type
    def __init__(self,
                 hidden_size=512,
                 num_attention_head:int=8,
                 # head_dim:int=512,
                 attn_dropout:float=0.0,
                 residual_dropout:float=0.0,
                 vocab_size:int =6400,
                 max_position_embeddings:int=32*1024,
                 feedforward_dim: int = None,  # FFN中间层维度，若为None则自动计算
                 # num_att_layer:int=10,
                 dropout:float=0.0001,
                 norm_eps=0.000001,
                 num_deep_layers:int=10,
                 use_moe:bool=False,
                 feedforward_act:str='silu',
                 inference_rope_scaling=False,
                         **kwargs
                 ):
        super().__init__(**kwargs)
        self.hidden_size=hidden_size
        self.num_attention_head=num_attention_head
        self.attn_dropout=attn_dropout
        self.residual_dropout=residual_dropout
        self.vocab_size=vocab_size
        self.max_position_embeddings=max_position_embeddings
        self.deep_layers=num_deep_layers
        self.dropout=dropout
        self.norm_eps=norm_eps
        self.feedforward_dim=feedforward_dim
        self.feedforward_act=feedforward_act
        self.use_moe=use_moe
        self.inference_rope_scaling=inference_rope_scaling
        # self.head_dim=head_dim



