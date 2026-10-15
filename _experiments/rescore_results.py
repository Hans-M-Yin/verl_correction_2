import argparse
import json
import asyncio
import random
from pathlib import Path
from tqdm import tqdm
import reward_function_loop as reward_loop
from wrong_caption_impact_eval import (
    load_experiment_dataset, filter_type_one_samples, 
    build_condition_summary, compute_paired_counts, 
    mcnemar_exact_pvalue, analyze_correct_wrong_caption_samples,
    GenerationRecord, WRONG_CAPTION_CONDITION, PLAIN_CONDITION,
    build_reward_extra_info, write_json, write_jsonl,
    # ngram_repetition_ratio # 确保已从原脚本导入
)
from reward_function_loop import ngram_repetition_ratio
def is_repeating(text: str) -> bool:
    """
    使用 reward_function_loop 中的逻辑判断文本是否在疯狂重复。
    如果 ngram_repetition_ratio 返回非 0 值（负值），则视为重复。
    """
    if not text or len(text) < 10:
        return False
    # 提取 predict_no_think 部分（去除 think 标签后的内容）
    content = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
    # 调用原逻辑：n=4, eps=0.85
    penalty = reward_loop.ngram_repetition_ratio(content, n=4, eps=0.85)
    return penalty != 0
async def rescore_records_async(records, sample_lookup, condition, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    async def _score_one(record):
        sample = sample_lookup[record["sample_id"]]
        extra_info = build_reward_extra_info(sample.extra_info, condition)
        async with semaphore:
            reward_output = await reward_loop.compute_score(
                data_source="rescore_experiment",
                solution_str=record["response"],
                ground_truth=record["ground_truth"],
                extra_info=extra_info,
                reward_router_address="",
                reward_model_tokenizer=None,
            )
        record.update({
            "score": float(reward_output["score"]),
            "acc_reward": float(reward_output["acc_reward"]),
            "correction_reward": float(reward_output["correction_reward"]),
            "format_reward": float(reward_output["format_reward"]),
            "repeat_penalty": float(reward_output["repeat_penalty"]),
        })
    tasks = [_score_one(r) for r in records]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Rescoring {condition}"):
        await f
    return records
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True, help="原始实验结果目录")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sample-size", type=int, default=1000, help="重评分采样条数")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    res_path = Path(args.results_dir)
    
    with open(res_path / "run_args.json", "r") as f:
        original_args = json.load(f)
    print(f"[*] 加载数据集: {original_args['dataset_path']}")
    dataset = load_experiment_dataset(original_args["dataset_path"], original_args["split"])
    samples = filter_type_one_samples(dataset, -1)
    sample_lookup = {s.sample_id: s for s in samples}
    # 加载旧数据
    def load_jsonl(p):
        with open(p, "r") as f:
            return {json.loads(line)["sample_id"]: json.loads(line) for line in f}
    wrong_map = load_jsonl(res_path / "wrong_caption_records.jsonl")
    plain_map = load_jsonl(res_path / "plain_records.jsonl")
    # 1. 数据清洗：剔除重复样本 (Pair-wise Filter)
    valid_ids = []
    for sid in wrong_map.keys():
        if sid not in plain_map: continue
        
        w_rep = is_repeating(wrong_map[sid]["response"])
        p_rep = is_repeating(plain_map[sid]["response"])
        
        if not w_rep and not p_rep:
            valid_ids.append(sid)
        else:
            # 可以在这里打印被剔除的 Sample ID 以便调试
            # print(f"[Skip] Sample {sid} filtered due to repetition.")
            pass
    print(f"[*] 过滤完成: 原始样本 {len(wrong_map)} 对, 有效非重复样本 {len(valid_ids)} 对。")
    # 2. 采样 1000 条
    if len(valid_ids) > args.sample_size:
        selected_ids = random.sample(valid_ids, args.sample_size)
        print(f"[*] 已从中随机采样 {args.sample_size} 条。")
    else:
        selected_ids = valid_ids
        print(f"[*] 有效样本不足 {args.sample_size}，将处理全部 {len(selected_ids)} 条。")
    wrong_to_score = [wrong_map[sid] for sid in selected_ids]
    plain_to_score = [plain_map[sid] for sid in selected_ids]
    # 3. 执行重评分
    reward_loop.init_openai_judge(args.api_key, args.base_url, args.judge_model)
    loop = asyncio.get_event_loop()
    
    print("[*] 正在对 Wrong Caption 组重新评分...")
    rescored_wrong = loop.run_until_complete(rescore_records_async(wrong_to_score, sample_lookup, WRONG_CAPTION_CONDITION, args.concurrency))
    
    print("[*] 正在对 Plain 组重新评分...")
    rescored_plain = loop.run_until_complete(rescore_records_async(plain_to_score, sample_lookup, PLAIN_CONDITION, args.concurrency))
    # 4. 统计结果并保存
    # (此处沿用 evaluate_wrong_caption.py 的分析逻辑，计算 Acc Drop, McNemar 等)
    def to_objs(records): return [GenerationRecord(**r) for r in records]
    
    w_objs = to_objs(rescored_wrong)
    p_objs = to_objs(rescored_plain)
    wrong_summary = build_condition_summary(w_objs)
    plain_summary = build_condition_summary(p_objs)
    paired_counts = compute_paired_counts(w_objs, p_objs)
    pvalue = mcnemar_exact_pvalue(paired_counts["wrong_only_correct"], paired_counts["plain_only_correct"])
    analysis = analyze_correct_wrong_caption_samples(w_objs, 6, 200)
    final_summary = {
        "sampled_size": len(selected_ids),
        "wrong_caption": wrong_summary,
        "plain": plain_summary,
        "accuracy_drop": plain_summary["accuracy"] - wrong_summary["accuracy"],
        "mcnemar_pvalue": pvalue,
        "analysis": analysis
    }
    # 保存
    new_dir = res_path / f"rescore_sampled_{args.judge_model}"
    new_dir.mkdir(exist_ok=True)
    write_json(new_dir / "summary.json", final_summary)
    print(f"\n[OK] 重评分报告已生成: {new_dir / 'summary.json'}")
    print(f"Accuracy Drop: {final_summary['accuracy_drop']:.4f}")
if __name__ == "__main__":
    main()