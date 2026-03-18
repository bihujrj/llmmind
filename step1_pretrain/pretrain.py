import os
import sys
import argparse
import time
import warnings
import math
from contextlib import nullcontext
from logging import Logger

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# 假设你的自定义模块都在这些路径下
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.dataset_pretrain import PretrainDataset
from model_def.llmconfig import LlmConfig
from model_def.attention import Attention   # 使用改进后的 Attention
from utils.train_tools import (
    get_lr, is_main_process, init_distributed_mode, lm_checkpoint,
    setup_seed, init_model, SkipBatchSampler, read_wandb_config
)

warnings.filterwarnings('ignore')

def train_epoch(epoch, model, loader, optimizer, scaler, args, wandb=None, start_step=0):
    model.train()
    total_steps = len(loader) + start_step
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将数据移到 GPU（如果可用）
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        # 动态调整学习率
        lr = get_lr(epoch * total_steps + step, args.epochs * total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（混合精度）
        with autocast(enabled=args.use_amp, dtype=torch.float16 if args.dtype == 'float16' else torch.bfloat16):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss + outputs.aux_loss   # 主损失 + MoE 辅助损失
            loss = loss / args.accumulation_steps    # 梯度累积平均

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积更新
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志打印
        if step % args.log_interval == 0 or step == total_steps:
            elapsed = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            aux_loss = outputs.aux_loss.item() if outputs.aux_loss is not None else 0.0
            logits_loss = current_loss - aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_remain = (elapsed / step) * (total_steps - step) / 60  # 剩余分钟

            Logger(
                f'Epoch [{epoch+1}/{args.epochs}] Step {step}/{total_steps} | '
                f'Loss: {current_loss:.4f} (logits: {logits_loss:.4f}, aux: {aux_loss:.4f}) | '
                f'LR: {current_lr:.8f} | ETA: {eta_remain:.1f} min'
            )
            if wandb:
                wandb.log({
                    'loss': current_loss,
                    'logits_loss': logits_loss,
                    'aux_loss': aux_loss,
                    'learning_rate': current_lr,
                    'step': step
                })

        # 保存 checkpoint
        if step % args.save_interval == 0 or step == total_steps:
            if is_main_process():
                # 保存模型权重
                suffix = '_moe' if args.use_moe else ''
                ckpt_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{suffix}.pth'
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 处理 torch.compile
                state_dict = raw_model.state_dict()
                torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckpt_path)

                # 保存完整训练状态（用于续训）
                lm_checkpoint(
                    model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    save_dir=args.save_dir,
                    wandb=wandb
                )
                del state_dict

def main():
    parser = argparse.ArgumentParser(description="MiniMind Pretraining (GPU Optimized)")
    parser.add_argument('--save_dir', type=str, default='../out', help='模型保存目录')
    parser.add_argument('--save_weight', default='pretrain', type=str, help='权重前缀')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')
    parser.add_argument('--device', type=str, default=None, help='设备，若为None则自动选择cuda或cpu')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='混合精度类型')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader工作线程数')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='梯度累积步数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=1000, help='保存间隔')
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='隐藏层数')
    parser.add_argument('--max_seq_len', type=int, default=340, help='最大序列长度')
    parser.add_argument('--use_moe', type=int, default=0, choices=[0,1], help='是否使用MoE')
    parser.add_argument('--data_path', type=str, default='../../../llm_data/pretrain_hq.jsonl', help='数据路径')
    parser.add_argument('--tokenizer_path', type=str, default='../model_ref', help='分词器路径')
    parser.add_argument('--from_weight', type=str, default='none', help='初始权重路径')
    parser.add_argument('--from_resume', type=int, default=0, choices=[0,1], help='是否从checkpoint恢复')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--wandb_project', type=str, default='LlmMind-Pretrain', help='wandb项目名')
    parser.add_argument('--use_compile', type=int, default=0, choices=[0,1], help='是否使用torch.compile')
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb API key')
    parser.add_argument('--force_gpu', action='store_true', help='如果没有GPU则报错')
    args = parser.parse_args()

    # python -m step1_pretrain.pretrain --tokenizer_path ./model_def --data_path ../../llm_data/pretrain_hq.jsonl

    # ----- 1. 初始化分布式环境 -----
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        if args.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)

    if args.force_gpu and device.type != 'cuda':
        raise RuntimeError('强制使用 GPU，但 CUDA 不可用！')
    args.device = device
    print(f'使用设备: {device}')

    # ----- 2. 设置随机种子 -----
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ----- 3. 创建保存目录 -----
    os.makedirs(args.save_dir, exist_ok=True)

    # ----- 4. 模型配置 -----
    config = LlmConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=8,                # 可从参数读取
        kv_heads=4,                            # GQA 的 kv head 数
        use_moe=bool(args.use_moe),
        max_seq_len=args.max_seq_len,
        attn_dropout=0.0,
        residual_dropout=0.0,
    )

    # ----- 5. 加载模型和分词器 -----
    model, tokenizer = init_model(
        config,
        from_weight=args.from_weight if args.from_weight != 'none' else None,
        tokenizer_path=args.tokenizer_path,
        device=device
    )

    # 确保模型参数在正确设备上
    assert all(p.device == device for p in model.parameters()), '模型参数未全部移动到指定设备！'

    if args.use_compile and device.type == 'cuda':
        model = torch.compile(model)
        print('torch.compile 已启用')

    # ----- 6. 数据集与 DataLoader -----
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # 使用 GPU 生成随机索引，减少 CPU 负担
    if sampler is None:  # 非分布式时使用随机采样
        # 一次性生成所有索引，并确保在 GPU 上生成后转回 CPU（因为 DataLoader 需要 CPU 索引）
        indices = torch.randperm(len(train_ds), device=device).cpu().tolist()
        batch_sampler = SkipBatchSampler(indices, args.batch_size, 0)
    else:
        batch_sampler = SkipBatchSampler(sampler, args.batch_size, 0)

    loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # ----- 7. 优化器与梯度缩放 -----
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler(enabled=(args.dtype == 'float16' and device.type == 'cuda'))

    # ----- 8. 恢复训练状态（续训）-----
    start_epoch, start_step = 0, 0
    if args.from_resume:
        ckpt_data = lm_checkpoint(config, weight=args.save_weight, save_dir=args.save_dir, load_only=True)
        if ckpt_data:
            model.load_state_dict(ckpt_data['model'])
            optimizer.load_state_dict(ckpt_data['optimizer'])
            scaler.load_state_dict(ckpt_data['scaler'])
            start_epoch = ckpt_data['epoch']
            start_step = ckpt_data.get('step', 0)
            print(f'从 epoch {start_epoch} step {start_step} 恢复训练')

    # ----- 9. 分布式封装 -----
    if dist.is_initialized():
        # 忽略某些 buffer（如 RoPE 缓存）
        model._ddp_params_and_buffers_to_ignore = {'cos_cached', 'sin_cached'}
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # ----- 10. WandB 初始化 -----
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wandb_lib
        if args.wandb_key is None:
            args.wandb_key = read_wandb_config()
        wandb_lib.login(key=args.wandb_key)
        run_name = f'pretrain_bs{args.batch_size}_lr{args.learning_rate}'
        wandb_lib.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb = wandb_lib

    # ----- 11. 训练循环 -----
    total_iters = len(loader)   # 每个 epoch 的迭代次数
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # 如果是续训且 epoch == start_epoch，需要跳过已处理的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        if skip > 0:
            # 更新 batch_sampler 以跳过前 skip 个 batch
            batch_sampler = SkipBatchSampler(sampler or indices, args.batch_size, skip)
            loader.batch_sampler = batch_sampler   # 直接替换 loader 的 batch_sampler
            print(f'Epoch {epoch+1}: 跳过前 {skip} 个 step')

        train_epoch(
            epoch=epoch,
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            wandb=wandb,
            start_step=skip
        )

        # 每个 epoch 结束后重置 start_step 为 0（下一个 epoch 从头开始）
        start_step = 0

    # ----- 12. 清理分布式进程 -----
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()