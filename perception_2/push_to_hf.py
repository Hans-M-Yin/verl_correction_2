from huggingface_hub import HfApi, create_repo, upload_folder

local_model_path = "merged_checkpoints/0414_base_run_30k_global_step_302"


repo_id = "hansQAQ/0414_base_run_30k_302"
create_repo(repo_id, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type="model",
)
