from huggingface_hub import HfApi, create_repo, upload_folder

local_model_path = "saved_models/0304_qwen2_5_vl_7b_mix_10K_no_kl_n8_170"


repo_id = "hansQAQ/0304_qwen2_5_vl_7b_mix_10K_no_kl_n8_170"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)
