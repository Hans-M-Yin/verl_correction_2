#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
import torch
import fire
from glob import glob
from collections import defaultdict
import os
import re


def main(fsdp_checkpoint_path, huggingface_model_path, output_path, dtype="float16"):
    """
    合并FSDP分片检查点到HuggingFace格式

    参数:
        fsdp_checkpoint_path: FSDP分片检查点路径
        huggingface_model_path: 原始HuggingFace模型路径（用于获取config和tokenizer）
        output_path: 输出路径
        dtype: 模型精度，可以是 "float16", "float32", "bfloat16"
    """
    # 导入分布式张量模块，解决 DTensor 加载问题
    import torch.distributed.tensor as dtensor

    state_dict = defaultdict(list)

    # 自动检测 world_size
    checkpoint_files = glob(f"{fsdp_checkpoint_path}/model_world_size_*_rank_*.pt")
    if not checkpoint_files:
        # 尝试其他可能的命名模式
        checkpoint_files = glob(f"{fsdp_checkpoint_path}/*_rank_*.pt")
        if not checkpoint_files:
            raise ValueError(f"在 {fsdp_checkpoint_path} 中未找到检查点文件")

    # 从文件名提取 world_size
    world_size = 0
    rank_files = {}

    for filepath in checkpoint_files:
        # 尝试多种文件名模式
        patterns = [
            r'model_world_size_(\d+)_rank_(\d+)\.pt',
            r'checkpoint_(\d+)_rank_(\d+)\.pt',
            r'rank_(\d+)_of_(\d+)\.pt',
            r'model_rank_(\d+)_world_size_(\d+)\.pt'
        ]

        match = None
        for pattern in patterns:
            match = re.search(pattern, os.path.basename(filepath))
            if match:
                break

        if match:
            if 'world_size' in filepath or 'of' in filepath:
                found_world_size = int(match.group(1))
                found_rank = int(match.group(2))
            else:
                found_world_size = int(match.group(2))
                found_rank = int(match.group(1))

            if world_size == 0:
                world_size = found_world_size
            elif world_size != found_world_size:
                print(f"警告: 检测到不一致的 world_size: {world_size} 和 {found_world_size}")

            rank_files[found_rank] = filepath

    if world_size == 0:
        # 如果没有匹配到模式，尝试从文件名数量推断
        world_size = len(checkpoint_files)
        print(f"警告: 无法从文件名确定world_size，假设为 {world_size}")
        rank_files = {i: checkpoint_files[i] for i in range(world_size)}

    print(f"检测到 world_size = {world_size}")

    # 按rank顺序加载
    for rank in sorted(rank_files.keys()):
        filepath = rank_files[rank]
        print(f'加载 rank {rank}: {filepath}')

        # 添加 weights_only=False 以支持 DTensor
        try:
            this_state_dict = torch.load(filepath, weights_only=False, map_location='cpu')
        except Exception as e:
            print(f"加载失败: {e}")
            raise

        for key, value in this_state_dict.items():
            # 检查是否是 DTensor，如果是则转换为本地张量
            if hasattr(value, 'to_local'):
                value = value.to_local()

            # 将值移动到CPU（如果尚未）
            if value.device != torch.device('cpu'):
                value = value.cpu()

            state_dict[key].append(value)

    # 检查每个key是否有正确数量的分片
    for key, values in list(state_dict.items()):
        if len(values) != world_size:
            print(f"警告: 参数 {key} 的分片数量 ({len(values)}) 不等于 world_size ({world_size})")

    # 合并所有分片
    merged_state_dict = {}
    for key in list(state_dict.keys()):
        values = state_dict[key]
        if len(values) > 1:
            # 尝试自动确定拼接维度
            # 对于线性层的权重，通常是二维的，在dim=0上拼接
            # 对于偏置，通常是一维的，也在dim=0上拼接
            # 对于卷积层权重，可能是四维的，需要根据具体情况处理

            # 检查所有分片的形状是否相同
            shapes = [v.shape for v in values]
            if all(s == shapes[0] for s in shapes):
                # 如果所有分片形状相同，可能是重复的分片（如LayerNorm参数）
                # 取第一个即可
                merged_state_dict[key] = values[0]
                print(f"参数 {key}: 所有分片形状相同 ({shapes[0]})，取第一个")
            else:
                # 尝试在dim=0上拼接
                try:
                    merged_state_dict[key] = torch.cat(values, dim=0)
                    print(f"参数 {key}: 在dim=0上拼接成功，形状 {merged_state_dict[key].shape}")
                except RuntimeError as e:
                    # 如果在dim=0上失败，尝试dim=1
                    try:
                        merged_state_dict[key] = torch.cat(values, dim=1)
                        print(f"参数 {key}: 在dim=1上拼接成功，形状 {merged_state_dict[key].shape}")
                    except RuntimeError:
                        print(f"警告: 无法拼接参数 {key}，形状: {shapes}")
                        # 保存所有分片供后续分析
                        for i, v in enumerate(values):
                            merged_state_dict[f"{key}_shard_{i}"] = v
        else:
            merged_state_dict[key] = values[0]

    # 加载配置
    print(f"从 {huggingface_model_path} 加载配置...")
    config = AutoConfig.from_pretrained(huggingface_model_path)

    # 设置dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"不支持的dtype: {dtype}")

    # 对于Qwen2.5-VL模型，使用AutoModel
    print("创建Qwen2.5-VL模型...")
    try:
        # 尝试使用AutoModel加载
        model = AutoModel.from_pretrained(
            huggingface_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载合并后的状态字典
        missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)

        if missing_keys:
            print(f"警告: 缺失的键 ({len(missing_keys)}个):")
            for i, key in enumerate(missing_keys[:10]):  # 只显示前10个
                print(f"  {key}")
            if len(missing_keys) > 10:
                print(f"  ... 还有 {len(missing_keys) - 10} 个")

        if unexpected_keys:
            print(f"警告: 意外的键 ({len(unexpected_keys)}个):")
            for i, key in enumerate(unexpected_keys[:10]):  # 只显示前10个
                print(f"  {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... 还有 {len(unexpected_keys) - 10} 个")

    except Exception as e:
        print(f"使用AutoModel加载失败: {e}")
        print("尝试直接从config创建模型...")

        # 尝试从config创建模型
        model = AutoModel.from_config(config, torch_dtype=torch_dtype)
        missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)

        if missing_keys:
            print(f"警告: 缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 意外的键: {unexpected_keys}")

    # 保存合并后的模型
    print(f"保存模型到 {output_path}...")
    model.save_pretrained(
        output_path,
        max_shard_size="10GB",
        safe_serialization=True
    )
    print(f"模型已保存到: {output_path}")

    # 保存tokenizer和processor
    print("保存tokenizer和processor...")
    try:
        # 对于VL模型，通常需要processor
        processor = AutoProcessor.from_pretrained(huggingface_model_path, trust_remote_code=True)
        processor.save_pretrained(output_path)
        print(f"Processor 已保存到: {output_path}")
    except:
        # 如果processor失败，只保存tokenizer
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        print(f"Tokenizer 已保存到: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)