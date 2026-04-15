from huggingface_hub import HfApi, create_repo, upload_folder
#
# local_model_path = "saved_models/0306_qwen2_5_vl_7b_mix_10K_no_correction_n8_280"
#
#
# repo_id = "hansQAQ/0306_qwen2_5_vl_7b_mix_10K_no_correction_n8_280"
# create_repo(repo_id, exist_ok=True)
#
# api = HfApi()
# api.upload_folder(
#     folder_path=local_model_path,
#     repo_id=repo_id,
#     repo_type="model",
# )

from huggingface_hub import HfApi, create_repo, login
import traceback

# 第1步：登录（确保已设置HF_TOKEN环境变量，或在此处填入token）
# 方式A（推荐）：通过环境变量HF_TOKEN自动认证
# 方式B：在代码中显式登录（如需，取消下一行注释）
# login(token="您的hf_xxx令牌")

# 第2步：配置仓库信息
repo_id = "hansQAQ/data_verl_correction"  # 请确保此用户名和仓库名正确
folder_to_upload = "./preprocessed_dataset"   # 本地文件夹路径
repo_type = "dataset"                         # 仓库类型
private = False                              # 是否设为私有仓库

# 第3步：创建HfApi实例
api = HfApi()

try:
    # 第4步：创建仓库（如果不存在）
    print(f"正在检查/创建仓库: {repo_id}")
    # create_repo 会处理仓库是否存在的情况，exist_ok=True 表示存在时不报错
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True
    )
    print("✅ 仓库已就绪。")

    # 第5步：上传文件夹
    print(f"开始上传文件夹: {folder_to_upload}")
    api.upload_folder(
        folder_path=folder_to_upload,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"✅ 文件夹上传完成！")
    print(f"仓库地址：https://huggingface.co/datasets/{repo_id}")

except Exception as e:
    print(f"❌ 操作失败: {e}")
    traceback.print_exc()  # 打印详细错误堆栈，便于进一步调试