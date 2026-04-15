import sys
sys.path.append('.')  # 确保可以导入model_merger模块

from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig  # 假设这个配置类存在

def main():
    config = ModelMergerConfig(
        operation="merge",  # 或者 "test"
        backend="fsdp",
        local_dir="checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor",  # 替换为你的FSDP检查点目录
        target_dir="saved_models/qwen2.5-vl-3b_verl_correction_data1_9k_210",  # 合并后模型保存的目录
        # test_hf_dir="/root/autodl-tmp/qwen2.5-vl-3b",  # 如果是测试，需要指定一个HF模型目录
        # hf_upload=False,  # 是否上传到HF Hub
        # 其他配置...
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()

if __name__ == "__main__":
    main()