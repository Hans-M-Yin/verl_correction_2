from transformers import AutoModel
import torch


def compare_models_transformers(model1_dir, model2_dir):
    """通过 Transformers 库加载并比较两个模型"""

    print(f"加载模型1: {model1_dir}")
    model1 = AutoModel.from_pretrained(model1_dir)
    model1.eval()

    print(f"加载模型2: {model2_dir}")
    model2 = AutoModel.from_pretrained(model2_dir)
    model2.eval()

    # 比较所有参数
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    print(f"\n模型1参数数量: {len(state_dict1)}")
    print(f"模型2参数数量: {len(state_dict2)}")

    # 检查键是否相同
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        print("✗ 模型结构不同！")
        return False

    # 比较数值
    all_same = True
    for name, param1 in state_dict1.items():
        param2 = state_dict2[name]

        if not torch.equal(param1, param2):
            diff = torch.abs(param1 - param2)
            max_diff = torch.max(diff).item()

            if max_diff > 1e-10:  # 检查显著差异
                print(f"✗ {name}: 不同 (max_diff={max_diff:.6e})")
                all_same = False
            else:
                print(f"✓ {name}: 相同 (微小数值差异)")

    if all_same:
        print("✓ 两个模型权重完全相同！")
    else:
        print("✗ 两个模型权重有差异")

    return all_same


# 使用示例
compare_models_transformers(
    "saved_models/test_qwen2.5-vl-3b_verl_correction_data1_9k_60",
    "saved_models/test_qwen2.5-vl-3b_verl_correction_data1_9k_210"
)