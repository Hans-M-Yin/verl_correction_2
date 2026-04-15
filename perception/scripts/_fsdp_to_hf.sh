python -m verl.model_merger merge \
 --backend fsdp \
 --local_dir "./checkpoints/verl_correction/0212_qwen2_5_vl_3b_data_mix_12k_correction_prompt/global_step_90/actor" \
 --target_dir "./saved_models/0219_qwen2_5_vl_3b_mix_12k_90"

 python -m verl.model_merger merge \
 --backend fsdp \
 --local_dir "./checkpoints/verl_correction/0212_qwen2_5_vl_3b_data_mix_12k_correction_prompt/global_step_135/actor" \
 --target_dir "./saved_models/0219_qwen2_5_vl_3b_mix_12k_135"

 python -m verl.model_merger merge \
 --backend fsdp \
 --local_dir "./checkpoints/verl_correction/0212_qwen2_5_vl_3b_data_mix_12k_correction_prompt/global_step_180/actor" \
 --target_dir "./saved_models/0219_qwen2_5_vl_3b_mix_12k_180"

 python -m verl.model_merger merge \
 --backend fsdp \
 --local_dir "./checkpoints/verl_correction/0212_qwen2_5_vl_3b_data_mix_12k_correction_prompt/global_step_249/actor" \
 --target_dir "./saved_models/0219_qwen2_5_vl_3b_mix_12k_249"