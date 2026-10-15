import json
import os
import asyncio
import aiohttp
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer
import re
import logging
from openai import OpenAI
import os

FORMAT_COEF = float(os.getenv("FORMAT_COEF", 0.4))
CORRECTION_COEF = float(os.getenv("CORRECTION_COEF",0.2))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY",0.8))


logger = logging.getLogger(__name__)
test_ip = "localhost:18903"
model_name = "qwen3-vl-8b"
# model_name = "qwen3-vl-30b-a3b"
async def re_hard_match(answer: str):
    if not answer:
        return None
    s = answer.strip()
    # 1. 明确的指示词 + 字母 (最高优先级)
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
    punct_pattern = re.compile(r'(?i)\b([A-F])(?=[\.\)\]：:）】\s]|$)')
    m = punct_pattern.findall(s)
    if m: return m[-1].upper()
    # 5. 独立字母（全文本只有一个字母的情况）
    pure_letter = re.findall(r'(?i)\b([A-F])\b', s)
    if len(set(pure_letter)) == 1:
        return pure_letter[0].upper()
    return answer


async def chat_complete(router_address: str, chat_complete_request: dict):
    url = f"http://{router_address}/v1/chat/completions"
    try:
        timeout = aiohttp.ClientTimeout(total=90)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(url, json=chat_complete_request) as resp:
            output = await resp.text()
            output = json.loads(output)
            return ChatCompletion(**output)
    except Exception as e:
        logger.warning(f"Chat Failed! {e}")
        raise e
    finally:
        await session.close()

async def llm_as_judge(data_source: str, answer_text: str, ground_truth: str, extra_info,    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer):
    # 4. Evaluate correctness using LLM judge
    question_text = extra_info.get("question", "") if extra_info else ""

    if not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0

    system_prompt = (
        "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
        "provided standard answer, given a specific question.\n"
        "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
        # "standard answer. \n"
        "standard answer. However, ignore the units difference (for example, the model's answer miss the units) when evaluation.\n"
        "Do not "
        'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
        "any explanation or other text."
    )

    user_prompt = (
        f"I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
        f"answer is correct.\n\n"
        f"---\n"
        f"**Example 1:**\n"
        f"[Question]: How long does the course maintain?\n"
        f"[Standard Answer]: 30 mins.\n"
        f"[Model's Answer]: 30.\n"
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
    # print(user_prompt)
    try:
        chat_complete_request = {
            "model":model_name,
            "messages":[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "seed":32768,
            "repetition_penalty":1.0,
            "presence_penalty":2.0,
            "top_p":1.0,
            "top_k":40,
            "temperature":1.0,  # Lower temperature for more deterministic judgement
        }
        chat_response = await chat_complete(router_address=reward_router_address, chat_complete_request=chat_complete_request)
        response = chat_response.choices[0].message.content.strip()
        # print(f"##### {response} {answer_text} {ground_truth}")
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


async def verify_logic_consistency(data_source: str, think_process: str, extra_info: dict, model_answer: str, reward_router_address: str):
    """
    Revised logic: Judge performs CoT analysis first, then infers the answer.
    Input includes the full modified caption and reasoning process.
    """
    wrong_caption = extra_info.get("caption_modified", "")
    question = extra_info.get("question", "")
    
    system_prompt = (
        "You are a logic consistency auditor examining a model's output. \n"
        "Your task is to determine which choice or specific answer the provided [Model Response] is logically leading to. \n"
        "Task Requirements:\n"
        "1. First, analyze the [Model Response] step-by-step to see what facts it identifies and how it handles potential conflicts. \n"
        "2. Then, conclude what final answer the response supports. If the reasoning is contradictory, incoherent, or reaches no specific choice, state 'INCONSISTENT'.\n"
        """3. Definition of INCONSISTENT

We define INCONSISTENT as cases where the [Model Response] contains internal logical contradictions in its reasoning process. Specifically, INCONSISTENT refers only to situations where the model’s statements, intermediate conclusions, or final answer cannot simultaneously hold true under the same assumptions.

INCONSISTENT does NOT include:
(1) Numerical or calculation errors (e.g., arithmetic mistakes, incorrect substitution, or wrong final numeric result).
(2) Incorrect but self-consistent reasoning (i.e., the reasoning is wrong but does not contradict itself).
(3) Self-correction or reflection steps, where the model explicitly identifies and fixes a previous mistake.
(4) Changes of answer that are consistently justified by updated reasoning or reinterpretation of the question.

A response is considered INCONSISTENT only if there exists a contradiction such that:
- The model asserts A and not-A within the same reasoning chain without resolution, or
- The final conclusion directly contradicts earlier derived conclusions that were not invalidated or revised in a logically consistent manner.
- The reasoning process is incomplete, and it is impossible to infer which answer the model selected from the [Model Response].
Examples of INCONSISTENT:
- The model derives x = 12, and later concludes x = -6 without any justification or revision step.
- The model states that B inhibits A, and later assumes B promotes A while still relying on the original assumption.
- The model concludes y₁ < y₂ based on monotonicity, but later states y₁ > y₂ without changing assumptions.

Non-INCONSISTENT examples:
- The model makes an arithmetic mistake but follows a consistent reasoning path.
- The model corrects itself (e.g., “I made a mistake earlier; the correct value is …”) and proceeds consistently from the corrected value."""
        "4. Your response MUST follow this format: <thinking> Your brief analysis (limited in 300 words.) </thinking> <answer> The inferred final answer or choice label </answer>."
    )
    
    # Prepend the wrong caption to ensure the judge sees the same world as the model
    combined_response = f"{wrong_caption}{think_process}"
    
    user_prompt = (
        f"[Question]: {question}\n"
        f"[Model Response]: {combined_response}\n\n"
        "--- Please perform the consistency audit:"
    )

    try:
        chat_complete_request = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        chat_response = await chat_complete(router_address=reward_router_address, chat_complete_request=chat_complete_request)
        response_content = chat_response.choices[0].message.content
        print(f"########  {response_content}")
        # Parse the judge's conclusion
        answer_match = re.search(r"<answer>(.*?)</answer>", response_content, re.DOTALL | re.IGNORECASE)
        inferred_answer = answer_match.group(1).strip() if answer_match else response_content.strip()
        
        # Validation
        if "INCONSISTENT" in inferred_answer.upper():
            return "INCONSISTENT"
        
        # Extract option letter if present
        label_match = re.search(r"\b([A-F])\b", inferred_answer.upper())
        if label_match:
            return label_match.group(1)
            
        return inferred_answer
    except Exception as e:
        logger.warning(f"Consistency Logic Audit failed: {e}")
        return "ERROR"

def ngram_repetition_ratio(tokens, n, eps=0.9):
    """
    Very basic n-gram repetition check.
    """
    if isinstance(tokens, str):
        tokens = tokens.split()
    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]
    if len(ngrams) == 0:
        return 0.0

    unique_ngrams = set(ngrams)
    rep = 1.0 - len(unique_ngrams) / len(ngrams)

    return 0.0 if rep < eps else -rep

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    if len(solution_str) < 50:
        logger.warning(f"####Output: {solution_str}")
    acc_reward, format_reward, consistency_reward, repeat_penalty = 0, 0, 0, 0
    try:
        is_format_error = False
        if extra_info['type'] == 1:
            count_think_1 = solution_str.count("<think>")
            count_think_2 = solution_str.count("</think>")
            if count_think_2 != 1 or count_think_1 != 0:
                is_format_error = True
        else:

            count_think_1 = solution_str.count("<think>")
            count_think_2 = solution_str.count("</think>")
            if count_think_1 != 1 or count_think_2 != 1:
                is_format_error = True
        
        answer_text = ""

        predict_no_think = (
            solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
        )

        # Check <answer> tag format
        count_answer_1 = predict_no_think.count("<answer>")
        count_answer_2 = predict_no_think.count("</answer>")
        if count_answer_1 != 1 or count_answer_2 != 1:
            is_format_error = True

        answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            # No proper <answer> tags found - this is a format error
            is_format_error = True

            if "</think>" in solution_str:
                answer_text = solution_str.split("</think>")[-1]

            else:
                answer_text = solution_str.strip()
        answer_text = answer_text.strip()
        if not answer_text:
            is_format_error = True
            answer_text = solution_str.strip()  # Use full text as last resort
        # print(answer_text)
        # 1. Base Accuracy Reward
        re_match_answer = await re_hard_match(answer_text)
        re_match_gt = await re_hard_match(ground_truth)
        # print(f"有点高: {re_match_answer} | {re_match_gt}")
        if re_match_answer and re_match_gt:
            acc_reward = 1.0 if re_match_answer == re_match_gt else 0.0
            if acc_reward != 1.0:
                acc_reward = await llm_as_judge(data_source, answer_text, ground_truth, extra_info, reward_router_address, reward_model_tokenizer)
        else:
            acc_reward = await llm_as_judge(data_source, answer_text, ground_truth, extra_info, reward_router_address, reward_model_tokenizer)
        if is_format_error:
            print(is_format_error, solution_str)
        
        # 2. Consistency Reward (1.0 for Match, 0.0 for Mismatch)
        if extra_info['type'] == 1 and not is_format_error:
            think_process = solution_str
            if "</think>" in solution_str:
                if think_process.count("</think>") > 1:
                    pattern = r'</?(think|answer)>'
                    think_process = re.sub(pattern, '', think_process)
                think_process = extra_info['caption_modified'] + think_process[:think_process.find("</think>")]
            think_process = think_process.replace("<think>", "")
            think_process = think_process.replace("</think>", "")
            inferred_ans = await verify_logic_consistency(data_source, think_process, extra_info, re_match_answer, reward_router_address)
            print(f"#INFER : {inferred_ans}")
            if not isinstance(inferred_ans, str):
                inferred_ans = str(inferred_ans)
            # Use LLM similarity check if it's not a single letter
            is_consistent = False
            if inferred_ans != "INCONSISTENT" and inferred_ans != "ERROR":
                if len(inferred_ans) == 1 and len(re_match_answer) == 1:
                    is_consistent = (inferred_ans.upper() == re_match_answer.upper())
                else:
                    # Semantic check for open-ended consistency
                    is_consistent = (inferred_ans.lower() in answer_text.lower()) or (answer_text.lower() in inferred_ans.lower())
            
            if is_consistent:
                consistency_reward = 1.0
            else:
                consistency_reward = 0.0
                logger.warning(f"Inconsistency: Model said {re_match_answer}, Judge inferred {inferred_ans}")
        else:
            consistency_reward = 0.0

        format_reward = -1.0 if is_format_error else 0.0
        repeat_penalty = ngram_repetition_ratio(solution_str, 4)

        # Final Score logic:
        # Base (1.0) + Consistency Bonus (0.2 * acc) + Format Penalty
        final_score = (
            1.0 * acc_reward + 
            FORMAT_COEF * format_reward + 
            CORRECTION_COEF * acc_reward * consistency_reward + 
            REPETITION_PENALTY * repeat_penalty
        )

        if final_score < -0.5:
             logger.warning(f"Low reward: {final_score:.2f} | Acc: {acc_reward} | Cons: {consistency_reward} | {solution_str[:100]}...")

    except Exception as e:
        logger.error(f"Error in compute_score: {e}")
        final_score = 0
    
    return {
        "score": final_score, 
        "acc_reward": acc_reward, 
        "correction_reward": consistency_reward, 
        "format_reward": format_reward, 
        "repeat_penalty": -repeat_penalty
    }


if __name__ == "__main__":
#     extra_info = {
#         "type": 1,
#         "split": "train",
#         "idx": 0,
#         "answer": "A",
#         "question": "Which one has a higher hospital beds per 1 population? (A) New Jersey (B) Georgia",
#         "caption_correct": "图片展示三角形ABC，其中OB和OC分别为∠ABC和∠ACB的内角平分线，相交于点O，连接OA。图中清晰标记了点A、B、C、O的位置及连线结构。",
#         "caption_modified": "The image is a choropleth map of the United States showing the distribution of hospital beds per 1 population. The states are color-coded into different ranges: 0.8-1.2, 0.5-0.7, 0.2-0.4, 0.0-0.1, and N/A (not applicable). New Jersey is shaded in a medium pink, which corresponds to the range 0.0-0.1. Georgia is shaded in a lighter pink, which corresponds to the range 0.2-0.4.",
#         "modify_info": [["New Jersey is 0.2-0.4 range",'New Jersey  is in 0.0-0.1 range.']
#                         ],
#         "num_modify": 1,
#         "original_output": "Hello"
#     }
#     solution_str = """
# Reflecting on the image, it seems there was a misinterpretation. New Jersey is actually in the 0.2-0.4 range, while Georgia is in the 0.0-0.1 range. Therefore, New Jersey does not have a higher value than Georgia.
# The correct answer is A.</think>
# <answer>A</answer>"""
    # extra_info = {
    #     "type": 1,
    #     "split": "train",
    #     "idx": 0,
    #     "answer": "$$5$$",
    #     "question": "Given the quadratic function $$y=ax^{2}+bx+c$$, part of the corresponding values between the function $$y$$ and the independent variable $$x$$ are shown in the table:  If points $$A(x_{1},y_{1})$$ and $$B(x_{2},y_{2})$$ are on the graph of the function, then when $$1< x_{1}<2$$ and $$3< x_{2}<4$$, the relationship between $$y_{1}$$ and $$y_{2}$$ is ___.",
    #     "caption_modified": "<think>In the image: When x=0, y=4; when x=1, y=1; when x=2, y=0; when x=3, y=1; when x=4, y=4. The function is a quadratic function y=ax²+bx+c. The table provides five data points for the function. The question asks to compare y₁ and y₂ for points A(x₁,y₁) and B(x₂,y₂) on the graph, where 3< x₁<4 and 1< x₂<2. Wait, I may be mistaken about what I see in the image."

    # }
    # solution_str = "  Let's correct it: The correct interpretation is that we should be comparing y₁ and y₂ for points A(x₁,y₁) and B(x₂,y₂) where 1< x₁<2 and 3< x₂<4, not 3< x₁<4 and 1< x₂<2. Let's correct the reasoning process.\n\nLet's correct the reasoning process: The correct interpretation of the question is that we should be comparing y₁ and y₂ for points A(x₁,y₁) and B(x₂,y₂) where 1< x₁<2 and 3< x₂<4, not 3< x₁<4 and 1< x₂<2. This is a correction in the logic of the question, not in the data provided in the image. Let's correct the comparison.\n\nGiven the correct interpretation, we should compare y₁ and y₂ for points A(x₁,y₁) and B(x₂,y₂) where 1< x₁<2 and 3< x₂<4. From the table, we can see that the function values for x=1 and x=3 are y=1 and y=1, respectively. However, the function is a quadratic function, and we need to consider the direction of the parabola. Since the function values at x=1 and x=3 are the same, but the function is a quadratic with a positive leading coefficient (implying it opens upwards), the function will be decreasing in the interval (1, 2) and increasing in the interval (3, 4). Therefore, y₁ will be greater than y₂ in the interval 1< x₁<2 and 3< x₂<4.\n</think>\n<answer>y₁ > y₂</answer>"
    # # Simple Async runner
    test_json = "../rollouts_saved/0418_with_correction_low_kl_30k_augmented/train/425.jsonl"
    import json

    def load_jsonl(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                data.append(json.loads(line))
        return data
    test_list = load_jsonl(test_json)
    from tqdm import tqdm
    for i in tqdm(test_list):
        async def test():
            answer = await compute_score("test", i['output'], i['gts'], {
                "type": 1 if "assistant\n<think>" in i['input'] else 0,
                "split": "train",
                "idx": 0,
                "answer": i['gts'],
                "question": i['input'][i['input'].find("user\n") + 4:i['input'].find("\nassistant")],
                "caption_modified": i['input'][i['input'].find("\nassistant\n") + 11: ]
            }, "localhost:18903", None)
            print(f"Score: {answer}")
        
        asyncio.run(test())