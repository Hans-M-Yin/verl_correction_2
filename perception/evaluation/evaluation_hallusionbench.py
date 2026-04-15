"""
HallusionBench dataset structure and adaptations
------------------------------------------------
Hugging Face dataset: `lmms-lab/HallusionBench`

Observed schema for `default/image`:
- `question`: yes/no question about the provided image
- `gt_answer`: string label `1` or `0`
- `gt_answer_details`: textual explanation of the correct judgment
- `image`: single image column
- extra metadata: `category`, `subcategory`, `visual_input`, `set_id`, `figure_id`,
  `sample_note`, `question_id`, `filename`

Adaptations in this script:
- Use the `image` split because it is the multimodal subset that matches the existing
  perception evaluation pipeline.
- Add benchmark-specific instructions so the model's final answer is explicitly `Yes` or `No`.
- Preserve all non-image metadata for later analysis and remove the image object before dump.
"""

import os
import json
import tqdm
import argparse
from datasets import load_dataset, Image
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor
from codetiming import Timer


SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags. "
    "The final answer MUST be either Yes or No, and MUST BE put in \\boxed{} in <answer> </answer> tags."
)

ds_collections = {
    "HallusionBench_image": {
        "root": "lmms-lab/HallusionBench",
        "max_new_tokens": 4096,
        "min_new_tokens": 1,
        "split": "image"
    },
}


def evaluate(args):
    dataset = load_dataset(ds_collections[args.datasets]["root"])[ds_collections[args.datasets]["split"]]
    dataset = dataset.cast_column("image", Image(decode=True))
    inputs = []
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_num_seqs=512,
        max_model_len=20000,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=1,
        n=1,
        max_tokens=4096,
        skip_special_tokens=False,
    )

    for idx, data_item in tqdm.tqdm(enumerate(dataset)):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": data_item["image"],
                    },
                    {
                        "type": "text",
                        "text": data_item["question"] + " Answer the question with Yes or No only. " + SYSTEM_PROMPT,
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        })

    with Timer(name="vLLM Perf Timer", text="{name}: Cosume {minutes:.2f} minutes for generation.") as timer:
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(f"{timer.name}: {timer.last}")

    outputs = []
    for dataitem, output in zip(dataset, model_outputs):
        del dataitem["image"]

        outputs.append({
            **dataitem,
            "response": output.outputs[0].text
        })
    output_filepath = os.path.join(args.out_dir, args.filename)
    json.dump(outputs, open(output_filepath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Results saved to {}".format(output_filepath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--datasets", type=str, default="HallusionBench_image")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="tests")
    parser.add_argument("--filename", type=str, default="hallusionbench_test.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    evaluate(args)
