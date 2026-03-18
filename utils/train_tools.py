import os
import sys
__package__ = "trainer"

from model_def.llmmodel import LlmModel

# from logging import Logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer

from pathlib import Path
import os
import logging

def Logger(content):
    if is_main_process():
        print(content)

def init_distributed_mode():
    # 检查环境变量 RANK 是否存在，若不存在或为 -1 则表示当前不是分布式训练模式
    # RANK 是全局进程编号，在分布式训练中由启动工具分配，从 0 开始
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP（Distributed Data Parallel）模式，直接返回 0，表示使用单卡

    # 初始化分布式进程组
    # backend="nccl" 指定使用 NVIDIA Collective Communications Library (NCCL) 作为后端
    # NCCL 是 NVIDIA 专门为 GPU 设计的高效通信库，支持多卡多机通信，如 all-reduce, broadcast 等
    # 在 PyTorch 分布式训练中，通常使用 NCCL 来同步梯度，因为它针对 GPU 进行了深度优化
    dist.init_process_group(backend="nccl")

    # 获取当前进程在本机上的局部排名（local rank），由启动器设置环境变量 LOCAL_RANK
    # 例如在 torchrun 中，LOCAL_RANK 表示当前进程在单机多卡中的 GPU 索引
    local_rank = int(os.environ["LOCAL_RANK"])

    # 设置当前进程使用的 GPU 设备
    # 确保每个进程只绑定到对应的 GPU，避免资源冲突
    torch.cuda.set_device(local_rank)

    # 返回 local_rank，以便后续代码使用当前 GPU 设备索引
    return local_rank

def get_lr(current_step, total_steps, lr):
    #学习率,以cos速度下降，相比1/N,前期下降慢，后期下降快
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    语言模型检查点保存/加载函数
    当 model 不为 None 时：保存模型权重（half精度）和恢复所需的所有状态（优化器、epoch、step、wandb id等）
    当 model 为 None 时：从恢复文件中加载状态，返回包含恢复信息的字典
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 根据是否使用MoE构建文件名后缀
    moe_path = '_moe' if lm_config.use_moe else ''
    # 模型权重文件（仅模型参数，half精度）
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 恢复文件（包含优化器、epoch、step等完整训练状态）
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:  # 保存模式
        # 如果模型被 DistributedDataParallel 包装，获取原始模型
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 如果模型被 torch.compile 包装，获取原始模型（_orig_mod 属性）
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        # 获取模型状态字典
        state_dict = raw_model.state_dict()
        # 转换为 half 精度并移到 CPU 保存，减小文件大小
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 先保存到临时文件，再原子替换，防止保存过程中断导致文件损坏
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)  # 原子替换

        # 获取 wandb 运行 ID 用于恢复时继续记录
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):  # swanlab 风格
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:  # wandb 风格
                wandb_id = getattr(wandb, 'id', None)

        # 构建恢复数据字典
        resume_data = {
            'model': state_dict,                # 模型权重（half）
            'optimizer': optimizer.state_dict(), # 优化器状态
            'epoch': epoch,                      # 当前轮数
            'step': step,                         # 当前步数（全局步数）
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,  # 当前GPU数量
            'wandb_id': wandb_id                   # wandb运行ID
        }

        # 处理额外的可保存对象（如学习率调度器、梯度缩放器等）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):  # 如果对象有 state_dict 方法（如 lr_scheduler）
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:  # 直接保存值（如 scaler 的 state_dict 已经是字典）
                    resume_data[key] = value

        # 同样原子保存恢复文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 清理临时变量，释放内存（模型权重已保存，可删除原字典）
        del state_dict, resume_data
        torch.cuda.empty_cache()  # 可选，及时释放显存

    else:  # 加载模式
        if os.path.exists(resume_path):
            # 加载恢复文件到 CPU
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)          # 保存时的GPU数量
            current_ws = dist.get_world_size() if dist.is_initialized() else 1  # 当前GPU数量

            # 如果GPU数量发生变化，调整 step 以保持大致相同的训练进度
            if saved_ws != current_ws:
                # step 是全局步数（所有GPU总步数），当GPU数变化时，需要按比例调整
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')

            return ckp_data  # 返回恢复数据字典
        return None  # 恢复文件不存在，返回空

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model_def', save_dir='../out', device='cuda'):
    print(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = LlmModel(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


#SkipBatchSampler 是一个自定义的 PyTorch Sampler，它的核心功能是在生成批次（batch）时跳过开头的若干个 batch。这在某些场景下很有用，例如从某个检查点恢复训练时，希望跳过已经处理过的数据批次，或者进行调试时忽略前几个 batch。
#
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)




def read_wandb_config(config_rel_path="llm_data/config.txt", levels_up=3):
    """
    从当前文件所在目录向上 levels_up 级目录下的 config_rel_path 文件中读取 wandb 配置。
    返回一个字典，包含所有以 'wandb_' 开头的键值对（键名转换为小写）。
    """
    # 获取当前文件所在目录的绝对路径
    current_dir = Path(__file__).resolve().parent
    # 向上 levels_up 级
    target_dir = current_dir
    for _ in range(levels_up):
        target_dir = target_dir.parent
    config_path = target_dir / config_rel_path

    wandb_config = {}
    if not config_path.exists():
        logging.warning(f"配置文件不存在: {config_path}")
        return wandb_config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳过空行和注释（以#开头）
                if not line or line.startswith('#'):
                    continue
                # 尝试用 '=' 或 ':' 分隔
                if '=' in line:
                    key, value = line.split('=', 1)
                elif ':' in line:
                    key, value = line.split(':', 1)
                else:
                    continue  # 不符合格式的行跳过
                key = key.strip().lower()
                value = value.strip()
                # 只保留 wandb 相关的配置
                if key.startswith('wandb_'):
                    wandb_config[key] = value
    except Exception as e:
        logging.error(f"读取配置文件 {config_path} 时出错: {e}")

    return wandb_config

# # 示例使用
# if __name__ == "__main__":
#     # 读取配置
#     config = read_wandb_config()
#     wandb_key = config.get('wandb_key')  # 根据文件中的键名，可能是 wandb_key
#     if wandb_key:
#         print(wandb_key)
#         print("成功读取 wandb_key")
#         # 在这里可以设置环境变量或直接用于 wandb.login
#         # os.environ['WANDB_API_KEY'] = wandb_key
#         # 或者
#         # import wandb
#         # wandb.login(key=wandb_key)
#     else:
#         print("未找到 wandb_key 配置")
