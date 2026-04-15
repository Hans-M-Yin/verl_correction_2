
import torch
import torch.distributed as dist
from pathlib import Path
import os
import json
import hashlib


def load_consolidated_fsdp_model(checkpoint_dir):
    """加载并合并FSDP分片模型（修复版本）"""
    checkpoint_dir = Path(checkpoint_dir)

    # 查找所有模型分片文件
    model_shards = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_shards:
        raise ValueError(f"在 {checkpoint_dir} 中找不到模型分片文件")

    print(f"找到 {len(model_shards)} 个模型分片")

    # 按rank排序
    model_shards.sort(key=lambda x: int(x.name.split('rank_')[-1].split('.')[0]))

    # 逐步合并所有分片
    consolidated_state_dict = {}

    for shard_path in model_shards:
        print(f"加载分片: {shard_path.name}")

        try:
            # 方法1：尝试使用 weights_only=False
            shard_state_dict = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"方法1失败: {e}")

            try:
                # 方法2：尝试导入 DTensor 相关模块
                import torch.distributed.tensor as dt
                shard_state_dict = torch.load(shard_path, map_location="cpu", weights_only=False)
            except Exception as e2:
                print(f"方法2失败: {e2}")

                try:
                    # 方法3：使用 pickle 直接加载（不推荐，仅作最后手段）
                    import pickle
                    with open(shard_path, 'rb') as f:
                        shard_state_dict = pickle.load(f)
                except Exception as e3:
                    print(f"所有方法都失败: {e3}")
                    continue

        # 合并到总状态字典
        for key, value in shard_state_dict.items():
            # FSDP可能会添加前缀，需要处理
            if key.startswith('_fsdp_wrapped_module.'):
                key = key.replace('_fsdp_wrapped_module.', '')

            # 处理 DTensor 转换为普通 Tensor
            if hasattr(value, 'to_local'):
                value = value.to_local()
            if hasattr(value, 'full_tensor'):
                value = value.full_tensor()

            consolidated_state_dict[key] = value

    print(f"合并后参数总数: {len(consolidated_state_dict)}")
    return consolidated_state_dict


def compare_state_dicts(state_dict1, state_dict2, checkpoint_name1="模型1", checkpoint_name2="模型2"):
    """比较两个状态字典的详细差异"""
    print(f"\n3. 比较模型结构")
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        print("✗ 模型键名不同！")
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"  仅在{checkpoint_name1}中的键 ({len(only_in_1)}个): {list(only_in_1)[:3]}...")
        if only_in_2:
            print(f"  仅在{checkpoint_name2}中的键 ({len(only_in_2)}个): {list(only_in_2)[:3]}...")
        return False

    print(f"✓ 模型结构相同，共有 {len(keys1)} 个参数组")

    # 比较数值
    print(f"\n4. 比较参数数值")
    differences = []
    total_params = 0
    different_params = 0

    for key in sorted(keys1):
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        total_params += tensor1.numel()

        # 检查形状
        if tensor1.shape != tensor2.shape:
            print(f"✗ {key}: 形状不同 {tensor1.shape} vs {tensor2.shape}")
            differences.append((key, "shape_mismatch", tensor1.shape, tensor2.shape))
            continue

        # 检查数值
        if not torch.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8):
            diff = torch.abs(tensor1 - tensor2)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()

            different_params += tensor1.numel()
            differences.append((key, max_diff, mean_diff, tensor1.shape))

    # 输出结果
    diff_ratio = different_params / total_params if total_params > 0 else 0
    print(f"\n5. 比较结果:")
    print(f"   总参数数量: {total_params:,}")
    print(f"   不同参数数量: {different_params:,}")
    print(f"   差异比例: {diff_ratio * 100:.6f}%")

    if differences:
        # 分离形状不匹配和数值差异
        shape_diffs = [d for d in differences if d[1] == "shape_mismatch"]
        value_diffs = [d for d in differences if d[1] != "shape_mismatch"]

        if shape_diffs:
            print(f"\n形状不匹配的参数 ({len(shape_diffs)}个):")
            for i, (key, _, shape1, shape2) in enumerate(shape_diffs[:5]):
                print(f"  {i + 1:2d}. {key}: {shape1} vs {shape2}")

        if value_diffs:
            print(f"\n数值差异最大的参数 (前10个):")
            value_diffs.sort(key=lambda x: x[1], reverse=True)  # 按最大差异排序

            for i, (key, max_diff, mean_diff, shape) in enumerate(value_diffs[:10]):
                print(f"  {i + 1:2d}. {key:<50} shape={shape} max={max_diff:.2e} mean={mean_diff:.2e}")

        return False
    else:
        print("✓ 两个检查点完全相同！")
        return True


def compare_fsdp_checkpoints(checkpoint_dir1, checkpoint_dir2):
    """比较两个FSDP检查点"""
    print("=" * 60)
    print("FSDP模型比较工具")
    print("=" * 60)

    checkpoint_name1 = Path(checkpoint_dir1).name
    checkpoint_name2 = Path(checkpoint_dir2).name

    # 首先尝试最简单的比较方法
    if try_simple_comparison(checkpoint_dir1, checkpoint_dir2):
        return

    # 如果简单方法失败，使用完整方法
    print(f"\n1. 加载检查点1: {checkpoint_dir1}")
    state_dict1 = load_consolidated_fsdp_model(checkpoint_dir1)

    print(f"\n2. 加载检查点2: {checkpoint_dir2}")
    state_dict2 = load_consolidated_fsdp_model(checkpoint_dir2)

    # 比较状态字典
    return compare_state_dicts(state_dict1, state_dict2, checkpoint_name1, checkpoint_name2)


def try_simple_comparison(checkpoint_dir1, checkpoint_dir2):
    """尝试简单的比较方法"""
    checkpoint_dir1 = Path(checkpoint_dir1)
    checkpoint_dir2 = Path(checkpoint_dir2)

    # 检查是否有huggingface目录
    hf_dir1 = checkpoint_dir1 / "huggingface"
    hf_dir2 = checkpoint_dir2 / "huggingface"

    if hf_dir1.exists() and hf_dir2.exists():
        print("检测到huggingface目录，使用简单比较方法...")
        return compare_huggingface_dirs(hf_dir1, hf_dir2)

    # 检查文件哈希
    print("尝试文件哈希比较...")
    if compare_file_hashes(checkpoint_dir1, checkpoint_dir2):
        return True

    # 检查文件大小和修改时间
    model_files1 = list(checkpoint_dir1.glob("model_*.pt"))
    model_files2 = list(checkpoint_dir2.glob("model_*.pt"))

    if len(model_files1) != len(model_files2):
        print(f"分片数量不同: {len(model_files1)} vs {len(model_files2)}")
        return False

    # 比较文件大小
    for f1, f2 in zip(sorted(model_files1), sorted(model_files2)):
        size1 = f1.stat().st_size
        size2 = f2.stat().st_size
        if size1 != size2:
            print(f"文件大小不同: {f1.name} ({size1} bytes) vs {f2.name} ({size2} bytes)")
            return False

    print("✓ 文件大小相同，但需要进一步比较内容")
    return False


def compare_file_hashes(checkpoint_dir1, checkpoint_dir2):
    """比较文件MD5哈希"""
    checkpoint_dir1 = Path(checkpoint_dir1)
    checkpoint_dir2 = Path(checkpoint_dir2)

    # 获取所有相关文件
    all_files1 = list(checkpoint_dir1.glob("*.pt")) + list(checkpoint_dir1.glob("*.json"))
    all_files2 = list(checkpoint_dir2.glob("*.pt")) + list(checkpoint_dir2.glob("*.json"))

    # 计算文件哈希
    hashes1 = {}
    for file_path in all_files1:
        with open(file_path, 'rb') as f:
            hashes1[file_path.name] = hashlib.md5(f.read()).hexdigest()

    hashes2 = {}
    for file_path in all_files2:
        with open(file_path, 'rb') as f:
            hashes2[file_path.name] = hashlib.md5(f.read()).hexdigest()

    # 比较哈希
    all_files = set(hashes1.keys()) | set(hashes2.keys())
    different_files = []

    for filename in sorted(all_files):
        hash1 = hashes1.get(filename, "文件不存在")
        hash2 = hashes2.get(filename, "文件不存在")

        if hash1 == hash2:
            print(f"✓ {filename}: 哈希相同")
        else:
            print(f"✗ {filename}: 哈希不同 ({hash1} vs {hash2})")
            different_files.append(filename)

    if not different_files:
        print("✓ 所有文件完全相同！")
        return True
    else:
        print(f"✗ 有 {len(different_files)} 个文件不同")
        return False


def compare_huggingface_dirs(hf_dir1, hf_dir2):
    """比较huggingface目录"""
    from transformers import AutoModel
    import torch

    print("通过Transformers加载模型...")

    try:
        # 尝试加载模型
        model1 = AutoModel.from_pretrained(str(hf_dir1), torch_dtype=torch.float16)
        model2 = AutoModel.from_pretrained(str(hf_dir2), torch_dtype=torch.float16)

        # 获取状态字典
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()

        # 使用之前定义的比较函数
        result = compare_state_dicts(sd1, sd2, "HuggingFace模型1", "HuggingFace模型2")

        # 清理内存
        del model1, model2, sd1, sd2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"通过Transformers比较失败: {e}")
        return False


def analyze_checkpoint_structure(checkpoint_dir):
    """分析检查点结构"""
    checkpoint_dir = Path(checkpoint_dir)

    print(f"\n分析检查点结构: {checkpoint_dir}")
    print("文件列表:")
    for file in checkpoint_dir.iterdir():
        if file.is_file():
            size = file.stat().st_size
            print(f"  {file.name} - {size:,} bytes")
        else:
            print(f"  {file.name}/ (目录)")

    # 检查FSDP配置
    config_path = checkpoint_dir / "fsdp_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"FSDP配置: {json.dumps(config, indent=2)}")

    # 检查huggingface目录
    hf_dir = checkpoint_dir / "huggingface"
    if hf_dir.exists():
        print("HuggingFace目录内容:")
        for file in hf_dir.iterdir():
            if file.is_file():
                size = file.stat().st_size
                print(f"  {file.name} - {size:,} bytes")


def main():
    """主函数"""
    # 使用示例
    checkpoint1 = "/root/autodl-tmp/perception/verl/checkpoints/verl_correction/qwen2_5_vl_3b_data1_3k/global_step_60/actor"
    checkpoint2 = "/root/autodl-tmp/perception/verl/checkpoints/verl_correction/qwen2_5_vl_3b_data1_3k/global_step_210/actor"

    # 首先分析检查点结构
    analyze_checkpoint_structure(checkpoint1)
    analyze_checkpoint_structure(checkpoint2)

    # 然后进行比较
    print("\n" + "=" * 80)
    result = compare_fsdp_checkpoints(checkpoint1, checkpoint2)

    if result:
        print("\n🎉 两个检查点完全相同！")
    else:
        print("\n⚠️ 两个检查点存在差异")

    return result


if __name__ == "__main__":
    # 设置环境（如果需要）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # 运行比较
    main()