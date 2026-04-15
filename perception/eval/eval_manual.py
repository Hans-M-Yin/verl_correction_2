"""
Evaluate MathVista benchmark manually.
Use vllm to deploy and inference, faster!

"""
import datasets
import json
import re
import requests
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import base64
from PIL import Image
import os
from datetime import datetime
import logging
import uuid
import mimetypes
from tqdm import tqdm
import base64
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18903/v1",
    api_key="EMPTY"
)
judge_client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key="sk-pgdNpO2MAkEGybNtGGQPyoqIlSXadqcvQK8A0RNw1WxZhnfh"
)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MathVistaEvaluator:
    """MathVista评估器"""

    def __init__(self,
                 model_url: str = "http://localhost:18903/v1/completions",  # 修改为你的vLLM地址
                 judge_url: Optional[str] = None,  # 可选：LLM评估服务的地址
                 model_name: Optional[str] = None,
                 sampling_params: Optional[dict] = None,
                 use_judge_llm: bool = False,
                 max_retries: int = 3,
                 timeout: int = 60):
        """
        初始化评估器

        Args:
            model_url: 部署的vLLM模型API地址
            judge_url: LLM评估服务的API地址（用于软匹配）
            model_name: 模型名称
            use_judge_llm: 是否使用LLM进行软匹配评估
            max_retries: 最大重试次数
            timeout: 请求超时时间
        """
        self.model_url = model_url
        self.judge_url = judge_url
        self.model_name = model_name
        self.sampling_params = sampling_params
        self.use_judge_llm = use_judge_llm
        self.max_retries = max_retries
        self.timeout = timeout

        # 预编译常用的正则表达式
        # TODO
        self.answer_patterns = [
            r"(?:答案|answer|Answer)[\s:：]*([A-D0-9\.\-+\/]+)",  # 匹配答案：A/B/C/D 或数字
            r"(?:答案|answer|Answer)[\s:：]*['\"]?([^'\"]+)['\"]?[\s。\.]",  # 匹配带引号的答案
            r"[\s\n]*([A-D])[\s\n]*$",  # 纯选项
            r"[\s\n]*([\-+]?\d*\.?\d+)[\s\n]*$",  # 纯数字
        ]

        # 推理步骤标识
        self.reasoning_patterns = [
            r"(?:思考|推理|reasoning|Reasoning|think|Think)[\s:：]*",
            r"首先|然后|接着|最后|因为|所以|因此",
            r"(?:step|Step)[\s\d]*[：:]\s*"
        ]

    def load_dataset(self, dataset_path: str) -> datasets.Dataset:
        """加载MathVista数据集"""
        # try:
        #     with open(dataset_path, 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        #
        #     logger.info(f"成功加载数据集，共 {len(data)} 个样本")
        #     return data
        # except Exception as e:
        #     logger.error(f"加载数据集失败: {e}")
        #     raise
        dataset = datasets.load_dataset(dataset_path, split='testmini')
        return dataset

    def encode_image(self, image_input):
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

    def build_prompt(self, item) -> List:
        """
        构建请求prompt

        注意：这里需要根据你的模型进行修改
        如果你的模型需要特殊的图片编码格式，请修改此函数
        """
        prompt_text = f"Answer the following question. You should think the question step by step first in <think></think> and finally provide the answer with few words or single character/number in <answer></answer>. \nQuestion:{item['question']}\n"
        image_url = self.encode_image(item['decoded_image'])
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_url}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]

        # 如果没有图片或编码失败，只发送文本
        return messages

    def infer(self, item, sampling_params, **kwargs) -> str:
        """
        调用模型进行推理

        Args:
            question: 问题文本
            image_path: 图片路径

        Returns:
            模型输出文本
        """
        messages = self.build_prompt(item)

        response = client.chat.completions.create(
            messages=messages,
            **sampling_params
        )
        response = response.choices[0].message.content
        return response

    def is_reasoning_model(self, response: str) -> bool:
        """判断模型输出是否包含推理步骤"""
        for pattern in self.reasoning_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False

    def extract_answer_hard(self, response: str) -> Tuple[str, bool]:
        """
        使用硬匹配提取答案

        Args:
            response: 模型输出

        Returns:
            Tuple[提取的答案, 是否成功提取]
        """
        if "<answer>" in response:
            answer = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if len(answer) > 0:
                answer = answer[0]
                return answer, True
        # 首先尝试匹配明确的答案模式
        for pattern in self.answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:  # 确保提取到非空字符串
                    return extracted, True

        # 如果没有匹配到明确模式，尝试提取最后一个数字或选项
        # 先查找A-D选项
        options_match = re.findall(r'[A-D]', response, re.IGNORECASE)
        if options_match:
            return options_match[-1].upper(), True

        # 再查找数字
        numbers_match = re.findall(r'[-+]?\d*\.?\d+', response)
        if numbers_match:
            return numbers_match[-1], True

        # 如果都没有，返回整个响应的最后一段
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) < 100:  # 避免过长的文本
                return line, True

        return response, False

    def judge_with_llm(self, model_answer: str, item):
        """
        使用LLM判断答案是否正确（软匹配）

        注意：需要部署一个评估LLM，或者使用API
        这里是一个示例实现
        """

        # 构建评估prompt
        judge_prompt = f"""
Below I will provide you with a question and a model's output. Your task is to judge whether the model's answer is correct.

Question: {item['question']}
Ground Truth: {item['answer']}
Model's output: {model_answer}

You should only output a word CORRECT/INCORRECT based on the above information. DO NOT output other words.
        """
        messages = [
            {
                "role": "user",
                "content": judge_prompt
            }
        ]
        response = judge_client.chat.completions.create(
            messages=messages,
            model="gpt-5-mini"
        )
        response = response.choices[0].message.content
        response = response.strip()
        if "INCORRECT" in response:
            return False

        return True

    def evaluate_answer(self, model_response: str, item) -> Tuple[bool, Dict[str, Any]]:
        """
        评估单个答案

        Args:
            model_response: 模型原始输出
            ground_truth: 标准答案

        Returns:
            Tuple[是否正确, 评估详情]
        """
        ground_truth = item['answer']
        details = {
            "raw_response": model_response,
            "has_reasoning": self.is_reasoning_model(model_response),
            "extracted_answer": "",
            "hard_match": False,
            "llm_judge": False,
            "correct": False
        }

        # 第一步：尝试硬匹配提取答案
        extracted_answer, success = self.extract_answer_hard(model_response)
        details["extracted_answer"] = extracted_answer
        details["hard_match"] = success

        if success:
            # 进行直接比较
            # 首先进行精确匹配
            if str(extracted_answer).strip() == str(ground_truth).strip():
                details["correct"] = True
                return True, details

            # 尝试标准化后比较
            def normalize(text: str) -> str:
                text = str(text).lower().strip()
                # 移除多余空格和标点
                text = re.sub(r'[^\w\s\.\-]', '', text)
                text = re.sub(r'\s+', ' ', text)
                return text

            if normalize(extracted_answer) == normalize(ground_truth):
                details["correct"] = True
                return True, details

            # 尝试数值比较
            try:
                # 如果都是数字，比较数值
                extracted_num = float(extracted_answer)
                gt_num = float(ground_truth)
                if abs(extracted_num - gt_num) < 0.0001:  # 容忍浮点误差
                    details["correct"] = True
                    return True, details
            except ValueError:
                pass
        # print(f"Response:{model_response} Ground Truth:{item['answer']} ")

        # 第二步：如果硬匹配失败且启用了LLM评估，使用LLM判断
        if not details["correct"] and self.use_judge_llm:
            details["llm_judge"] = True
            details["correct"] = self.judge_with_llm(model_response, item)

        return details["correct"], details

    def evaluate_dataset(self,
                         dataset
                         ) -> Dict[str, Any]:
        """
        评估整个数据集

        Args:
            dataset: 数据集列表
            image_dir: 图片目录路径

        Returns:
            评估结果
        """
        results = []
        correct_count = 0
        total_count = len(dataset)

        for item in tqdm(dataset):

            # 调用模型推理
            try:
                response = self.infer(item, self.sampling_params)
            except Exception as e:
                logger.error(f"推理失败: {e}")
                response = ""

            # 评估答案
            is_correct, details = self.evaluate_answer(response, item)
            print(f"Response:{response} Ground Truth:{item['answer']} Correct:{is_correct}")
            if is_correct:
                correct_count += 1

            # 保存样本结果
            sample_result = {
                "sample_id": item['pid'],
                "question": item['question'],
                # "image": image_filename,
                "ground_truth": item['answer'],
                "model_response": response,
                "is_correct": is_correct,
                "details": details
            }
            results.append(sample_result)

        # 计算总体指标
        accuracy = correct_count / total_count if total_count > 0 else 0

        evaluation_result = {
            "model_name": self.model_name,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark": "MathVista",
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": accuracy,
            "config": {
                "model_url": self.model_url,
                "use_llm_judge": self.use_judge_llm,
                "judge_url": self.judge_url
            },
            "results": results
        }

        return evaluation_result

    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果到JSON文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"结果已保存到: {output_path}")

            # 同时保存一份简化版本
            summary = {
                "model_name": results["model_name"],
                "evaluation_date": results["evaluation_date"],
                "benchmark": results["benchmark"],
                "total_samples": results["total_samples"],
                "correct_samples": results["correct_samples"],
                "accuracy": results["accuracy"]
            }

            summary_path = output_path.replace('.json', '_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            logger.info(
                f"评估完成！准确率: {results['accuracy']:.4f} ({results['correct_samples']}/{results['total_samples']})")

        except Exception as e:
            logger.error(f"保存结果失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="MathVista多模态模型评估")
    parser.add_argument("--model_name", type=str,
                        default="multimodal-model",
                        help="模型名称")
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=int, default=0.8)

    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--judge_url", type=str,
                        default=None,
                        help="LLM评估服务地址（用于软匹配）")
    parser.add_argument("--output", type=str,
                        default="./results/evaluation_results.json",
                        help="输出结果文件路径")

    args = parser.parse_args()
    sampling_params = {
        "model": args.model_name,
        "temperature": args.temperature,
        "seed": args.seed,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens
    }
    # 创建评估器
    evaluator = MathVistaEvaluator(
        model_name=args.model_name,
        use_judge_llm=True,
        sampling_params=sampling_params
    )

    # 加载数据集
    dataset = evaluator.load_dataset(args.dataset)

    # 如果指定了最大样本数，截取部分数据用于测试

    # 评估数据集
    results = evaluator.evaluate_dataset(dataset)

    # 保存结果
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()