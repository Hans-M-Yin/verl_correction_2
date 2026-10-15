"""Qwen3 reward function variant without literal <think> tag requirements.

Qwen3 reasoning tokens are not reliably emitted as the textual ``<think>`` and
``</think>`` tags used by the Qwen2.5 training recipe.  This module reuses the
existing judge/correction implementation, but its format reward validates only
one explicit ``<answer>...</answer>`` pair anywhere in the response.
"""

from __future__ import annotations

import re
import time

from perception_2 import reward_function_loop_qwen3 as base


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer,
):
    """Score a Qwen3 rollout without imposing textual reasoning-tag syntax."""
    reward_start_time = time.perf_counter()
    correction_reward_time_s = 0.0
    correction_reward_eligible = 0
    acc_reward = 0.0
    format_reward = 0.0
    correction_reward = 0.0
    repeat_penalty = 0.0
    final_score = 0.0

    try:
        # Deliberately do not inspect <think> or </think>.  The only formatting
        # contract for Qwen3 is one final answer enclosed in answer tags.
        answer_open_count = solution_str.count("<answer>")
        answer_close_count = solution_str.count("</answer>")
        is_format_error = answer_open_count != 1 or answer_close_count != 1

        answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            # Preserve the existing fallback behaviour for answer correctness,
            # while applying the format penalty above.
            answer_text = solution_str.strip()
            is_format_error = True
        if not answer_text:
            answer_text = solution_str.strip()
            is_format_error = True

        re_match_answer = await base.re_hard_match(answer_text)
        if re_match_answer is not None:
            re_hard_match_ground_truth = await base.re_hard_match(ground_truth)
            if re_hard_match_ground_truth is not None:
                acc_reward = float(re_match_answer.lower().strip() == re_hard_match_ground_truth.lower().strip())
            else:
                acc_reward = await base.llm_as_judge(
                    data_source, answer_text, ground_truth, extra_info,
                    reward_router_address, reward_model_tokenizer,
                )
            if acc_reward < 1:
                acc_reward = float(re_match_answer.lower().strip() == ground_truth.lower().strip())
        else:
            acc_reward = await base.llm_as_judge(
                data_source, answer_text, ground_truth, extra_info,
                reward_router_address, reward_model_tokenizer,
            )

        format_reward = -1.0 if is_format_error else 0.0

        if extra_info.get("type") == 1 and base.CORRECTION_COEF > 0.01:
            correction_reward_eligible = 1
            correction_start_time = time.perf_counter()
            try:
                correction_count, failure_count, state, failure_reason = await base.compute_correction_reward(
                    data_source, solution_str, ground_truth, extra_info,
                    reward_router_address, reward_model_tokenizer,
                )
            finally:
                correction_reward_time_s = time.perf_counter() - correction_start_time

            if state:
                correction_reward = correction_count / extra_info["num_modify"]
                correction_reward = min(correction_reward, 1.0)
            elif correction_count < 0:
                correction_reward = -1.0
                base.logger.warning("Correction judged SPURIOUS: %s", failure_reason)
            else:
                base.logger.warning(
                    "Encounter failure when calculating correction reward: %s",
                    failure_reason or "unknown reason",
                )

        # Unlike the previous implementation, this considers the complete
        # response, because Qwen3 may use non-textual/native reasoning tokens.
        repeat_penalty = base.ngram_repetition_ratio(solution_str, 4)
        final_score = (
            acc_reward
            + base.FORMAT_COEF * format_reward
            + base.CORRECTION_COEF * acc_reward * correction_reward
            + base.REPETITION_PENALTY * repeat_penalty
        )
    except Exception as exc:
        base.logger.warning("Qwen3 no-think reward computation failed: %s", exc)
        final_score = 0.0

    reward_total_time_s = time.perf_counter() - reward_start_time
    return {
        "score": final_score,
        "acc_reward": acc_reward,
        "correction_reward": correction_reward,
        "format_reward": format_reward,
        "repeat_penalty": -repeat_penalty,
        "correction_reward_time_s": correction_reward_time_s,
        "correction_reward_eligible": correction_reward_eligible,
        "other_reward_time_s": max(reward_total_time_s - correction_reward_time_s, 0.0),
        "reward_total_time_s": reward_total_time_s,
    }
