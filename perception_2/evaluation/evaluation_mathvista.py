import argparse
import json
import os
import random

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor
from vllm import LLM, SamplingParams


import json
import uuid
import mimetypes
from datetime import datetime
import os


def encode_image(image_input):
    # print(image_input)

    if hasattr(image_input, 'save'):
        buffered = BytesIO()
        image_input.save(buffered, format=image_input.format if image_input.format else 'PNG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    image_path = image_input
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')




import base64
import requests
from PIL import Image
from io import BytesIO


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path

import logging
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # 使用 'w' 模式覆盖原有日志文件，实现清空效果
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


ds_collections = {
    'MathVista_testmini': {
        'root': 'AI4Math/MathVista',
        'max_new_tokens': 4096,
        'min_new_tokens': 1,
        'split': 'testmini'
    },
}

SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags. "
)


def evaluate_chat_model(args):
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(ds_collections[ds_name]['root'])[ds_collections[ds_name]['split']]

        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            data_item['query'] = data_item['query']
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": data_item['decoded_image']
                        },
                        {
                            "type": "text",
                            "text": data_item['query'] + " " + SYSTEM_PROMPT,
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

        # sampling_params = SamplingParams(temperature=0.01, top_p=0.001, top_k=1, max_tokens=4096,
        sampling_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            n=1,
            max_tokens=4096,
            skip_special_tokens=False,
        )
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item['decoded_image']
            data_item['response'] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            pid = data_item['pid']
            temp[pid] = data_item

        print(f'Evaluating {ds_name} ...')
        results_file = args.filename
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MathVista_testmini')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--filename', type=str, default='mathvista.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

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
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model(args)