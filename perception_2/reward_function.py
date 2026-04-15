import re
import io
import logging
import os
import random
import re
import time
import requests
from openai import OpenAI
from PIL import Image


logger = logging.getLogger(__name__)

openai_api_key = "EMPTY"
openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://222.29.51.14:18903/v1")
model_name = "qwen3-vl-30b"
# openai_api_key = "sk-dsNGokuZOsnm8gKru7chFbhhZnVy8Jx9IVvull1duVPIBiPD"
# openai_api_base = "https://yunwu.ai/v1"
# model_name = "gemini-2.5-flash-nothinking"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

async def llm_as_judge(data_source: str, answer_text: str, ground_truth: str, extra_info):
    # 4. Evaluate correctness using LLM judge
    question_text = extra_info.get("question", "") if extra_info else ""

    if not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0

    system_prompt = (
        "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
        "provided standard answer, given a specific question.\n"
        "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
        "standard answer.\n"
        'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
        "any explanation or other text."
    )

    user_prompt = (
        f"I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
        f"answer is correct.\n\n"
        f"---\n"
        f"**Example 1:**\n"
        f"[Question]: Is the countertop tan or blue?\n"
        f"[Standard Answer]: The countertop is tan.\n"
        f"[Model's Answer]: tan\n"
        f"[Your Judgement]: CORRECT\n"
        f"---\n"
        f"**Example 2:**\n"
        f"[Question]: Is the man phone both blue and closed?\n"
        f"[Standard Answer]: Yes, the man phone is both blue and closed.\n"
        f"[Model's Answer]: No.\n"
        f"[Your Judgement]: INCORRECT\n"
        f"---\n"
        f"**Task:**\n"
        f"[Question]: {question_text}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model's Answer]: {answer_text}\n"
        f"[Your Judgement]:"
    )
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.1,  # Lower temperature for more deterministic judgement
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f" [WARNING] Chat completion request failed: {e}")
        return 0.0
    if re.search(r"\bCORRECT\b", response, re.IGNORECASE):
        acc_reward = 1.0
    elif re.search(r"\bINCORRECT\b", response, re.IGNORECASE):
        acc_reward = 0.0
    else:
        logger.warning(
            f" [WARNING] Judgement format error. Expected 'CORRECT' or 'INCORRECT'.\n"
            f"Response: '{response}'\n"
            f"Model Answer: '{answer_text}'\n"
            f"Ground Truth: '{ground_truth}'"
        )
        acc_reward = 0.0
    return acc_reward

def re_hard_match(answer: str):
    if not answer:
        return None
    s = answer.strip()

    # 1. 明确的指示词 + 字母 (最高优先级)
    # 增加 "综上"、"最后" 等词汇
    phrase_pattern = re.compile(
        r'(?i)(?:综上所述|总结|最终|答案|选择|choose|is|are|是|为|答案[:：]|选项)[:：\s]*([A-F])\b'
    )
    m = phrase_pattern.findall(s)
    if m: return m[-1].upper() # 取最后一个出现的明确答案

    # 2. Markdown 加粗形式 **A** (高优先级)
    md_pattern = re.compile(r'(?i)\*\*\s*([A-F])\s*\*\*')
    m = md_pattern.findall(s)
    if m: return m[-1].upper()

    # 3. 括号/标点包围
    bracket_pattern = re.compile(
        r'(?:(?<=\()|(?<=\[)|(?<=\{)|(?<=（)|(?<=【))\s*([A-Fa-f])\s*(?=[\)\]\}）】])'
    )
    m = bracket_pattern.findall(s)
    if m: return m[-1].upper()

    # 4. 字母 + 标点/空白 (如 "C.", "D:", "A ")
    # 注意避免匹配到单词开头，\b 很重要
    punct_pattern = re.compile(r'(?i)\b([A-F])(?=[\.\)\]：:）】\s]|$)')
    m = punct_pattern.findall(s)
    if m: return m[-1].upper()

    # 5. 独立字母（全文本只有一个字母的情况）
    # 如果全文只有 1 个 A-F 的字母，那基本就是它了
    pure_letter = re.findall(r'(?i)\b([A-F])\b', s)
    if len(set(pure_letter)) == 1:
        return pure_letter[0].upper()
    return None

def compute_correction_reward(data_source, solution_str, ground_truth, extra_info):
    # 调LLM-as-judge感觉太花时间了啊

    modify_info = extra_info['modify_info']
    num_modify = extra_info['num_modify']

    modify_info_str = "\n".join([str(idx) + f". {k[0]}" for idx, k in enumerate(modify_info)])

    if not client or not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0
    system_prompt = """
    You are an expert evaluator. Your task is to determine whether the model's output contains specific information, i.e., to detect whether the target text is present in the model's output text.    

    1. Make judgments based on the semantic meaning of the model's output. As long as the meaning is consistent with the specific information provided by the user, that information is considered to be present. 
    2. The user will provide you with several pieces of specific information, and you need to search the model's output to determine whether each corresponding piece of content appears.
    3. Strictly follow the format below: According to the order of the specific information provided by the user, output whether each corresponding piece appears in sequence. Output YES if it appears, and NO if it does not. Separate multiple outputs with spaces, and add a index before each answer.

    For example:
    [Model's Output]: <think>In the image there is a sequence of green apples in the left but not the right. Instead the girl is in the right side of the image. We can indicate that the girl is going to eat the apple. So the final answer is C.</think><answer>C</answer>
    [Specific Information]:
    1. Apple is green.
    2. There is one green apple.
    2. The boy is in the right side.
    [Your Answer]:
    1.YES 2.NO 3.NO
    (Here is the reason, and you don't need to output the reason: In the provided output, the model points out that the apple is green. But the model outputs the wrong number of apples. Thirdly, in the image it's a boy but not a girl. So the ideal determination is YES NO YES.)  
    """
    user_prompt = f"""
    I will provide a model's output, and several specific information. You must determine if the model's output contains these information respectively.
    Remember to follow the instruction and the format! 
    [Model's Output]: {solution_str}
    [Specific Information]: 
    {modify_info_str}
    [Your Answer]:
    """
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=42,
            temperature=0.1,  # Lower temperature for more deterministic judgement
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f" [WARNING] Chat completion request failed: {e}")
        return 0, 0, False
    if "boxed{" in response:
        response = response[response.find("boxed{"):]
    # print(response)
    response_temp = response
    if "(" in response:
        response_temp = re.sub(r'\([^)]*\)', '', response)
    correction_count = response_temp.lower().count("yes")
    failure_count = response_temp.lower().count("no")
    if correction_count + failure_count != num_modify:
        if correction_count + failure_count / 2 == num_modify:
            logger.warning(
                f"Perhaps misalignment correction output: Requires: {num_modify}, received: {response} ({correction_count} + {failure_count})")
            correction_count = int(correction_count / 2)
            failure_count = int(failure_count / 2)
        else:
            logger.warning(
                f" [WARNING] No enough correction output. Requires: {num_modify}, received: {response} ({correction_count} + {failure_count})")
    return correction_count, failure_count, True

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    acc_reward, format_reward, correction_reward = 0, 0, 0
    try:
        """Compute the reward score."""
        is_format_error = False

        # 1. Check <think> tag format
        count_think_1 = solution_str.count("<think>")
        count_think_2 = solution_str.count("</think>")
        if count_think_1 != count_think_2:
            is_format_error = True

        # 2. Check vision tokens (skip this since tokenizer removes special tokens)
        # We'll use <tool_call> and <tool_response> instead to detect tool usage

        # 3. Extract answer text with multiple fallback strategies
        answer_text = ""

        # Strategy 1: Try to extract from <answer> tags first
        predict_no_think = (
            solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
        )

        # Check <answer> tag format
        count_answer_1 = predict_no_think.count("<answer>")
        count_answer_2 = predict_no_think.count("</answer>")
        if count_answer_1 != count_answer_2:
            is_format_error = True

        # Try to extract from <answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            # No proper <answer> tags found - this is a format error
            is_format_error = True

            if "</think>" in solution_str:
                # Remove any remaining tool-related tags and extract meaningful content
                answer_text = solution_str.split("</think>")[-1]
                # answer_text = remaining_content.strip()
            else:
                # Strategy 4: Use the entire solution_str as fallback
                answer_text = solution_str.strip()
        answer_text = answer_text.strip()
        if not answer_text:
            is_format_error = True
            answer_text = solution_str.strip()  # Use full text as last resort

        re_match_answer = re_hard_match(answer_text)
        # print(f"Hard match {re_match_answer}")
        if re_match_answer is not None:
            acc_reward = 1.0 if re_match_answer.lower().strip() == ground_truth.lower().strip() else 0.0
        else:
            acc_reward = llm_as_judge(data_source, answer_text, ground_truth, extra_info)
        format_reward = -1.0 if is_format_error else 0.0

        if is_format_error or not answer_text:
            logger.debug(
                f"Format issue detected:\n"
                f"Solution: {solution_str[:200]}...\n"
                f"Extracted answer: '{answer_text}'\n"
                f"Format error: {is_format_error}\n"
            )
        correction_reward = acc_reward
        if extra_info['type'] == 1:
            correction_count, failure_count, state = compute_correction_reward(data_source, solution_str, ground_truth, extra_info)
            if state:
                correction_reward = correction_count / extra_info['num_modify']
        # logger.info(f"{acc_reward} | {format_reward} | {correction_reward}")
        # Final weighted score
        final_score = 0.8 * acc_reward + 0.4 * format_reward + 0.8 * correction_reward
    except Exception as e:
        final_score = 0
    return {"score": final_score, "acc_reward": acc_reward, "format_reward": correction_reward}

if __name__ == "__main__":
    # 这个依然有点不可验证的感觉。用LLM-as-judge还是太主观了。
    # Qwen2.5-VL-7B肯定不能作为judge model，实在太弱了。
    extra_info =  {
        "type": 1,
        "split": "train",
        "idx": 0,
        "answer": "A",
        "question": "△ABC的两内角平分线OB、OC相交于点O，若∠A＝110°，则∠BOC＝（）",
        "caption_correct": "图片展示三角形ABC，其中OB和OC分别为∠ABC和∠ACB的内角平分线，相交于点O，连接OA。图中清晰标记了点A、B、C、O的位置及连线结构。",
        "caption_modified": "图片展示三角形ABC，其中OB和OC分别为∠ABC和∠ACB的内角平分线，并且相交于点A，连接OA。图中清晰标记了点A、B、C、O的位置及连线结构。",
        "modify_info": [["A在上方顶点，B在左下", "B在上方顶点，A在左下"], ['O位于三角形内部','O位于三角形外部'], ["最终是两条角平分线在O点相交", '两条角平分线在A点相交']],
        "num_modify": 3,
        "original_output": "Hello"
    }
    # print("你好！")
    solution_str = """好的。为了计算∠BOC，我们只需要知道∠OBC和∠OCB。现在让我在看一下图，等下，图片中A在上方，B在下方，并且原图中两条角平分线相交于O，而不是A点。假设∠A=110°，那么∠OBC+∠OCB=35°，所以∠BOC=180°-35°=145°。答案选A。</think> A选项应该是对的。"""
    compute_score("编的",solution_str,"A",extra_info)