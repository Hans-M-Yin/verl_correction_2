from collections import defaultdict
from copy import deepcopy

from verl import DataProto
import json
import re
import random
import math
from typing import Literal
import logging

logger = logging.getLogger(__name__)

triggers = None
def has_chinese(text):
    """判断字符串中是否包含中文字符"""
    pattern = re.compile(r'[\u4e00-\u9fff]')
    # print( bool(pattern.search(text)))
    return bool(pattern.search(text))


def init_triggers(triggers_file_path):
    global triggers
    with open(triggers_file_path, 'r', encoding='utf-8') as f:
        _triggers = json.load(f)
    triggers = _triggers
    print(" Initialize trigger done. ")


def sample_trigger(prompt):
    prompt = deepcopy(prompt)
    prompt_assistant = prompt[-1]
    assert triggers is not None
    language = "english"
    if isinstance(prompt_assistant, str):
        language = "chinese" if has_chinese(prompt_assistant) else "english"
        prompt_assistant += " " + random.choice(triggers[language])
    elif isinstance(prompt_assistant, dict):
        if 'content' in prompt_assistant.keys():
            if isinstance(prompt_assistant['content'], list):
                language = "chinese" if has_chinese(prompt_assistant['content'][-1]['text']) else "english"
                prompt_assistant['content'][-1]['text'] += " " + random.choice(triggers[language])
                # logger.warning(f"### Current :{prompt_assistant['content'][-1]['text']}")
            elif isinstance(prompt_assistant['content'], str):
                language = "chinese" if has_chinese(prompt_assistant['content']) else "english"
                prompt_assistant['content'] += " " + random.choice(triggers[language])
            else:
                raise ValueError("Not support type for assistant_prompt_content {}".format(type(prompt_assistant['content'])))
    else:
        raise ValueError("Not support type for assistant_prompt_content {}".format(type(prompt_assistant['content'])))
    prompt[-1] = prompt_assistant
    return prompt

def trigger_schedule_policy(
    global_step,
    total_step,
    start_ratio=1.0,
    final_rest=0.1,
    warmup_ratio=0.3,
):
    """
    Annealing scheduler: returns probability p(step).
    - p(0) = start_ratio
    - p(total_step) = final_rest
    - Early stage decays slowly, later stage faster.
    """
    if total_step <= 0:
        return start_ratio
    progress = global_step / total_step
    progress = min(max(progress, 0.0), 1.0)
    if progress <= warmup_ratio:
        prob = start_ratio
    else:
        t = (progress - warmup_ratio) / (1.0 - warmup_ratio)
        prob = final_rest + (start_ratio - final_rest) * math.exp(-5.0 * t)

    return prob


def add_correction_trigger(
        batch: DataProto,
        rollout_n: int,
        global_step: int,
        total_step: int,
        warmup_ratio: float,
        start_ratio: float,
        final_ratio: float,
        mode: Literal['all', 'part'],
        enable_schedule: bool
):
    assert len(batch) % rollout_n == 0, f"Repeated samples ({len(batch)}) can't be divided by rollout_n ({rollout_n})"
    num_samples = len(batch) // rollout_n
    trigger_prob = trigger_schedule_policy(global_step, total_step, start_ratio, final_ratio, warmup_ratio)
    new_batch = batch
    if mode == 'all':
        # Whether all rollout in one sample use triggers, or none uses triggers
        for i in range(num_samples):
            start_idx = i * rollout_n
            if new_batch.non_tensor_batch['extra_info'][start_idx]['type'] != 1:
                continue
            if enable_schedule and random.random() < trigger_prob:

                end_idx = start_idx + rollout_n
                # for j in range(start_idx, end_idx):
                #     logger.warning(f"{j} : {new_batch.non_tensor_batch['prompt'][j]}")

                for j in range(start_idx, end_idx):
                    # logger.warning(f"########## BEFORE {j} : {new_batch.non_tensor_batch['prompt'][j]}")
                    if 'prompt' in new_batch.non_tensor_batch:
                        new_batch.non_tensor_batch['prompt'][j] = sample_trigger(new_batch.non_tensor_batch['prompt'][j])
                    elif 'prompt' in new_batch.batch:
                        new_batch.batch['prompt'][j] = sample_trigger(new_batch.batch['prompt'][j])
                    if 'raw_prompt' in new_batch.non_tensor_batch:
                        new_batch.non_tensor_batch['raw_prompt'][j] = new_batch.non_tensor_batch['prompt'][j]
                    elif 'raw_prompt' in new_batch.batch:
                        new_batch.batch['raw_prompt'][j] = new_batch.batch['prompt'][j]
                    # logger.warning(f"########## AFTER {j} : {new_batch.non_tensor_batch['prompt'][j]}")

    elif mode == 'part':
        add_trigger_num_per_sample = int(rollout_n * trigger_prob)
        logger.warning(f"################### PER_SAMPLE{add_trigger_num_per_sample}")
        for i in range(num_samples):
            start_idx = i * rollout_n
            # end_idx = start_idx + rollout_n
            if new_batch.non_tensor_batch['extra_info'][start_idx]['type'] != 1:
                continue
            for j in range(start_idx, start_idx + add_trigger_num_per_sample):
                # logger.warning(f"########## {add_trigger_num_per_sample} BEFORE {j} : {new_batch.non_tensor_batch['prompt'][j]}")

                if 'prompt' in new_batch.non_tensor_batch:
                    new_batch.non_tensor_batch['prompt'][j] = sample_trigger(new_batch.non_tensor_batch['prompt'][j])
                elif 'prompt' in new_batch.batch:
                    new_batch.batch['prompt'][j] = sample_trigger(new_batch.batch['prompt'][j])
                if 'raw_prompt' in new_batch.non_tensor_batch:
                    new_batch.non_tensor_batch['raw_prompt'][j] = new_batch.non_tensor_batch['prompt'][j]
                elif 'raw_prompt' in new_batch.batch:
                    new_batch.batch['raw_prompt'][j] = new_batch.batch['prompt'][j]
                # logger.warning(f"########## {add_trigger_num_per_sample} AFTER {j} : {new_batch.non_tensor_batch['prompt'][j]}")

    else:
        raise NotImplementedError

    return new_batch

