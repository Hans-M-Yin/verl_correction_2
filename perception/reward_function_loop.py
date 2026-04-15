
import json
import os
import asyncio
import aiohttp
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer
import re
import logging
from openai import OpenAI

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
    return None


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
        logger.warning(f"Chat Failed!",  exc_info=e)
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
        logger.warning(f" [WARNING] Chat completion request failed", exc_info=e)
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


async def compute_correction_reward(data_source, solution_str, ground_truth, extra_info,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer
 ):
    # logger.warning("SUPER POWER OF HANS MARCUS YIN.")
    think_process = solution_str
    if "</think>" in solution_str:
        if think_process.count("</think>") > 1:
            pattern = r'</?(think|answer)>'
            think_process = re.sub(pattern, '', think_process)
        think_process = think_process[:think_process.find("</think>")]
    think_process = extra_info['caption_modified'] + ' ' + think_process
    modify_info = extra_info['modify_info']
    num_modify = extra_info['num_modify']

    modify_info_str = "\n ".join([f"<info{str(idx + 1)}>" + f" Wrong: {k[1]} → Correct: {k[0]} " + f"</info{str(idx + 1)}>" for idx, k in enumerate(modify_info)])

    if not reward_router_address:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0
    system_prompt = """
Reasoning: low
You are an expert evaluator.

Your task is to determine whether the Given Text shows COMPLETE and EXPLICIT self-correction.

For each Information Item, check two STRICT conditions:

Condition 1 (Wrong Mention):
- The Given Text MUST explicitly state the Wrong Description (or an equivalent sentence).
- If the Wrong Description does NOT appear in the text, output NO.

Condition 2 (Correction Statement):
- After stating the wrong description, the text MUST explicitly state the Correct Description.
- The correction must be clearly written as a factual statement.

IMPORTANT STRICT RULES:

- If only the correct description appears but the wrong description does NOT appear, output NO.
- If the text only implies the wrong description is wrong, output NO.
- If the text is logically inconsistent, output NO.
- Do NOT infer, guess, or assume anything. Only judge based on exact text content.

Output YES ONLY if BOTH:
1) The wrong description appears in the text.
2) The correct description appears after that.

Output NO otherwise.

**Output FORMAT**:

<think> step by step reasoning process for each information item.</think><info1>YES/NO</info1><info2>YES/NO</info2>...


**Example:**
[Given Text]: "In the center of the image there is a red apple. The boy is on the right, holding a fresh banana. Wait I may make mistake about the image, actually the apple is green, not red."
[Information Items]:
<info1> Wrong: The apple is red. → Correct: The apple is green. </info1>
<info2> Wrong: There is a sequence of apples. → Correct: There is one apple. </info2>
<info3> Wrong: The girl is in the right side. → Correct: The boy is in the right side. </info3>

Your response:

"<think> I will think step by step first.
1. info1: The text explicitly denies "red" and states "green". → YES
2. info2: The text does not mention "one apple" or state the quantity of apples. → NO
3. info3: The text mentions "boy", which affirms the correct component, BUT the text does not deny the wrong part directly (e.g., 'there is a boy but not a girl'"), so even the correct part is stated, I will give a NO. → NO
So I will give the final response now.
</think>
<info1>YES</info1><info2>NO</info2><info3>NO</info3>"
    """
    user_prompt = f"""
I will provide you with a given text paragraph, and several specific information. For each piece of specific information, you must determine if the text paragraph contains the information.
Remember to follow the instruction. You MUST follow the output FORMAT! 
[Given text]: "{think_process}"
[Information Items]: 
"{modify_info_str}"

Your response:
"""
    try:
        chat_complete_request = {
            "model":model_name,
            "messages":[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = await chat_complete(router_address=reward_router_address, chat_complete_request=chat_complete_request)
        response = response.choices[0].message.content
        # logger.warning(f"################ {response}")
    except Exception as e:
        logger.warning("Failure when computing correction reward")
        return 0, 0, False
    response_temp = response
    if "<info1>" in response:

        response_temp = response[response.rfind("<info1>"):]

    correction_count = 0
    failure_count = 0
    correct_correction_judgment_format = True
    for i in range(num_modify):

        cnt = re.findall( rf'<info{str(i + 1)}>(.*?)</info{str(i + 1)}>', response_temp, re.DOTALL)
        if len(cnt) < 1:
            correct_correction_judgment_format = False
            logger.warning(f"######### Not qualified correction judgement: {response}")

            break
        else:
            cnt_yes_count = cnt[-1].lower().count("yes")
            # if cnt_yes_count > 0:
                # logger.warning(f"######### [DEBUG] {response.replace('\n',' ')} ( ||||  {modify_info_str.replace('\n', ' ')}  |||| {solution_str.replace('\n', ' ')})")
            cnt_no_count = cnt[-1].lower().count("no")
            if cnt_yes_count > 1 or cnt_no_count > 1 or cnt_yes_count * cnt_no_count > 0:
                if 'yes' in cnt[-1].lower().strip()[:-4]:
                    cnt_no_count = 0
                    cnt_yes_count = 1
                elif 'no' in cnt[-1].lower().strip()[:-4]:
                    cnt_no_count = 1
                    cnt_yes_count = 0
                else:
                    logger.warning(f"######### Unsatisfied correction response:{response_temp.replace('\n', ' ')} ( current: {cnt} YES: {cnt_yes_count} NO: {cnt_no_count}")
            correction_count += cnt_yes_count
            failure_count += cnt_no_count
    if not correct_correction_judgment_format:
        correction_count = response_temp.lower().count("yes")
        failure_count = response_temp.lower().count("no")
    # logger.warning(f"########### {solution_str.replace("\n"," ")} || TEMPLATE {modify_info_str.replace("\n", "    ")} || ANSWER || {response.replace("\n"," ")} || CORRECT: {correction_count}, FAILURE: {failure_count}")
    if correction_count + failure_count != num_modify:
        if (correction_count + failure_count) / 2 == num_modify:
            logger.warning(f"Perhaps misalignment correction output: Requires: {num_modify}({modify_info_str.replace("\n", "    ")}), received: {response_temp.replace("\n", " ")} ({correction_count} + {failure_count})")
            correction_count = int(correction_count / 2)
            failure_count = int(failure_count / 2)
        else:
            logger.warning(
                f" [WARNING] No enough correction output. Requires: {num_modify}({modify_info_str.replace("\n", "    ")}), received: {response_temp.replace("\n", " ")} ({correction_count} + {failure_count})")
        return 0,0, False
    return correction_count, failure_count, True

def ngram_repetition_ratio(tokens, n, eps=0.85):
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
    acc_reward, format_reward, correction_reward, repeat_penalty = 0, 0, 0, 0
    try:
    # if True:
        """Compute the reward score."""
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

        # Try to extract from <answer> tags
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

        re_match_answer = await re_hard_match(answer_text)
        # print(re_match_answer, ' ###### MATCH ANSWER')
        # print(f"Hard match {re_match_answer}")
        if re_match_answer is not None:
            re_hard_match_ground_truth = await re_hard_match(ground_truth)
            # print(re_hard_match_ground_truth, ' ###### MATCH ANSWER')

            if re_hard_match_ground_truth is not None:
                acc_reward = 1.0 if re_match_answer.lower().strip() == re_hard_match_ground_truth.lower().strip() else 0.0
            else:
                acc_reward = await llm_as_judge(data_source, answer_text, ground_truth, extra_info,
                                                reward_router_address, reward_model_tokenizer)
            if acc_reward < 1:
                acc_reward = 1.0 if re_match_answer.lower().strip() == ground_truth.lower().strip() else 0.0

        else:
            acc_reward = await llm_as_judge(data_source, answer_text, ground_truth, extra_info, reward_router_address, reward_model_tokenizer)
        format_reward = -1.0 if is_format_error else 0.0

        if is_format_error or not answer_text:
            logger.debug(
                f"Format issue detected:\n"
                f"Solution: {solution_str[:200]}...\n"
                f"Extracted answer: '{answer_text}'\n"
                f"Format error: {is_format_error}\n"
            )
        # correction_reward = acc_reward
    # NOTICE!!!!!
        correction_reward = 0.0
        # if extra_info['type'] == 1:
        #     correction_count, failure_count, state, = await compute_correction_reward(data_source, solution_str, ground_truth, extra_info, reward_router_address, reward_model_tokenizer)
        #     if state:
        #         correction_reward = correction_count / extra_info['num_modify']
        #         if correction_reward > 1.0:
        #             logger.warning(f"Bigger correction reward than excepted:{correction_reward}")
        #             correction_reward = 1.0
        #     else:
        #         logger.warning("Encounter failure when calculating correction reward ")
        # else:
        #     correction_reward = 0.0
        repeat_penalty = ngram_repetition_ratio(predict_no_think, 4)
        if repeat_penalty < -0.1:
            final_score = 1 * acc_reward + 0.4 * format_reward + 0.2 * acc_reward * correction_reward + 0.8 * repeat_penalty
        else:
            final_score = 1 * acc_reward + 0.4 * format_reward + 0.2 * acc_reward * correction_reward + 0.8 * repeat_penalty
        if final_score < -0.5:
            logger.warning(f"Reward less: {final_score} | {acc_reward} | {format_reward} | repeat_penalty: {repeat_penalty} | {solution_str.replace('\n', ' ')}")

    except Exception as e:
        print(e)
        final_score = 0
    return {"score": final_score, "acc_reward": acc_reward, "correction_reward": correction_reward, "format_reward": format_reward, "repeat_penalty": -repeat_penalty}


if __name__ == "__main__":
    extra_info = {
        "type": 1,
        "split": "train",
        "idx": 0,
        "answer": "A",
        "question": "Which one has a higher hospital beds per 1 population? (A) New Jersey (B) Georgia",
        "caption_correct": "图片展示三角形ABC，其中OB和OC分别为∠ABC和∠ACB的内角平分线，相交于点O，连接OA。图中清晰标记了点A、B、C、O的位置及连线结构。",
        "caption_modified": "The image is a choropleth map of the United States showing the distribution of hospital beds per 1 population. The states are color-coded into different ranges: 0.8-1.2, 0.5-0.7, 0.2-0.4, 0.0-0.1, and N/A (not applicable). New Jersey is shaded in a medium pink, which corresponds to the range 0.0-0.1. Georgia is shaded in a lighter pink, which corresponds to the range 0.2-0.4.",
        "modify_info": [["New Jersey is 0.2-0.4 range",'New Jersey  is in 0.0-0.1 range.']
                        ],
        "num_modify": 1,
        "original_output": "Hello"
    }

    # solution_str = """<think> The color wheel in the image is a representation of the Hue-Saturation subspace. The saturation coefficient is the amount of color present in a hue. The more saturated a color is, the more it deviates from the center of the circle. Therefore, the color that is closest to the center of the circle will have the smallest saturation coefficient. In the image, color A is the closest to the center, so it has the smallest saturation coefficient. </think>\n<answer> A </answer>"""
    # 使用 asyncio.run() 运行异步函数
    solution_str = """
Reflecting on the image, it seems there was a misinterpretation. New Jersey is actually in the 0.2-0.4 range, while Georgia is in the 0.0-0.1 range. Therefore, New Jersey does not have a higher value than Georgia. Correcting the initial thought, the answer is actually that New Jersey does not have a higher value than Georgia.

Correcting the answer: The initial interpretation was incorrect. New Jersey is in the 0.2-0.4 range, and Georgia is in the 0.0-0.1 range. Therefore, New Jersey does not have a higher value than Georgia. The correct answer is B.

Original answer: B</think>
<answer>B</answer>"""
    answer = asyncio.run(compute_score("编的", solution_str, "E. (b).", extra_info, test_ip, None))
    print(f"Score: {answer}")