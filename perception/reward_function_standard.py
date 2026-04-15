
import json
import os
import asyncio
import aiohttp
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer
import re
import logging

logger = logging.getLogger(__name__)
test_ip = "172.17.0.2:18903"
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
    # print("正在突击欧金金")
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(url, json=chat_complete_request) as resp:
            output = await resp.text()
            output = json.loads(output)
            return ChatCompletion(**output)
    except Exception as e:
        logger.warning(f"Chat Failed!")
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


async def compute_correction_reward(data_source, solution_str, ground_truth, extra_info,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer
 ):
    think_process = solution_str
    if "</think>" in solution_str:
        if think_process.count("</think>") > 1:
            pattern = r'</?(think|answer)>'
            think_process = re.sub(pattern, '', think_process)
        think_process = think_process[:think_process.find("</think>")]

    modify_info = extra_info['modify_info']
    num_modify = extra_info['num_modify']

    modify_info_str = "\n".join([str(idx + 1) + f". ## {k[0]} ##" for idx, k in enumerate(modify_info)])

    if not reward_router_address:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0
    system_prompt = """
You are an expert evaluator. Your task is to determine whether the given paragraph contains specific information.

Instructions:
1. Place YES/NO based on whether the piece of specific information is present semantically. Only when the text fully mentions the specific information explictly, you will place a YES. 
2. Judge each piece of information item by item. Each item you ONLY need to check whether the text clearly contains the specific information.
3. For each specific information you must output ONLY ONE SINGLE YES/NO. Do NOT repeat the same item of specific information, a single specific information matches only ONE output.
4. Output MUST be in this exact format: <think>your think process, how you determine each specific information</think><answer>YES/NO YES/NO (totally the same number with pieces of specific information) </answer>
5. If the given text seems containing heavy repetition, please output REPEAT.

Example:
[Given text]: "The apple is green. A girl is on the right."
[Specific Information]:
"1. ## Apple is green and there is also a banana. ##
2. ## There is one green apple. ##
3. ## The boy is in the right side. The boy is looking at his phone. ##" 

Your response:
<think>
1. ## Apple is green and there is also a banana ##: Text says apple is green → YES
2. ## There is one green apple. ##: Text mentions a sequence of apples, not ONE apple → NO  
3. ## The boy is in the right side. The boy is looking at his phone. ##: Text says girl, not boy, and text doesn't mention phone. → NO
</think>
<answer> YES NO NO </answer>
    """
    user_prompt = f"""
I will provide you with a given text paragraph, and several specific information. For each piece of specific information, you must determine if the text paragraph contains the information.
Remember to follow the instruction and the format! your evaluation must match the specific information each by each.
[Given text]: "{think_process}"
[Specific Information]: 
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
            "seed":32768,
            "repetition_penalty":1.0,
            "presence_penalty":2.0,
            "top_p":1.0,
            "top_k":40,
            "temperature":1.0,  # Lower temperature for more deterministic judgement
        }
        response = await chat_complete(router_address=reward_router_address, chat_complete_request=chat_complete_request)
        response = response.choices[0].message.content
    except Exception as e:
        logger.warning("Failure when computing correction reward")
        return 0, 0, False
    response_temp = response
    # repeat punishment
    if "REPEAT" in response_temp:
        logger.warning(f"Detect REPEAT: ###{think_process.replace("\n", " ")}###")
        return -num_modify, 0, True
    if "<answer>" in response and "</answer>" in response:
        response_temp = re.findall(r'<answer>(.*?)</answer>', response)
        if len(response_temp) > 1:
            logger.warning(f"Multiple answers found: {response_temp} | {response.replace("\n", " ")}")
        response_temp = response_temp[-1]
    else:
        logger.warning(f"Wrong format when parsing correction reward: {response.replace("\n", " ")}")
    correction_count = response_temp.lower().count("yes")
    failure_count = response_temp.lower().count("no")
    # logger.warning(f"########### {solution_str.replace("\n"," ")} || TEMPLATE {modify_info_str.replace("\n", "    ")} || ANSWER || {response.replace("\n"," ")}")
    if correction_count + failure_count != num_modify:
        if (correction_count + failure_count) / 2 == num_modify:
            logger.warning(f"Perhaps misalignment correction output: Requires: {num_modify}({modify_info_str.replace("\n", "    ")}), received: {response.replace("\n", " ")} ({correction_count} + {failure_count})")
            correction_count = int(correction_count / 2)
            failure_count = int(failure_count / 2)
        else:
            logger.warning(
                f" [WARNING] No enough correction output. Requires: {num_modify}({modify_info_str.replace("\n", "    ")}), received: {response.replace("\n", " ")} ({correction_count} + {failure_count})")
    return correction_count, failure_count, True

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    # logger.warning(f"&&&&{solution_str}&&&&")
    acc_reward, format_reward, correction_reward = 0, 0, 0
    try:
        """Compute the reward score."""
        is_format_error = False
        if extra_info['type'] == 2:
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
                answer_text = solution_str.split("</think>")[-1]

            else:
                answer_text = solution_str.strip()
        answer_text = answer_text.strip()
        if not answer_text:
            is_format_error = True
            answer_text = solution_str.strip()  # Use full text as last resort

        re_match_answer = await re_hard_match(answer_text)
        # print(f"Hard match {re_match_answer}")
        if re_match_answer is not None:
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
        correction_reward = acc_reward
        if extra_info['type'] == 1:
            correction_count, failure_count, state, = await compute_correction_reward(data_source, solution_str, ground_truth, extra_info, reward_router_address, reward_model_tokenizer)
            # correction_count= 0
            state = True
            if state:
                correction_reward = correction_count / extra_info['num_modify']
            else:
                logger.warning("Encounter failure when calculating correction reward ")
        # logger.info(f"{acc_reward} | {format_reward} | {correction_reward}")
        # Final weighted score
        final_score = 0.8 * acc_reward + 0.4 * format_reward + 0.4 * correction_reward
    except Exception as e:
        final_score = 0
    return {"score": final_score, "acc_reward": acc_reward, "correction_reward": correction_reward, "format_reward": format_reward}


if __name__ == "__main__":
    extra_info = {
        "type": 1,
        "split": "train",
        "idx": 0,
        "answer": "A",
        "question": "△ABC的两内角平分线OB、OC相交于点O，若∠A＝110°，则∠BOC＝（）",
        "caption_correct": "图片展示三角形ABC，其中OB和OC分别为∠ABC和∠ACB的内角平分线，相交于点O，连接OA。图中清晰标记了点A、B、C、O的位置及连线结构。",
        "caption_modified": "图片展示三角形ABC，其中OB和OC分别为∠ABC和∠ACB的内角平分线，并且相交于点A，连接OA。图中清晰标记了点A、B、C、O的位置及连线结构。",
        "modify_info": [[" Triangle ABC is isosceles with AB = AC."], [' Vertices are arranged with A at the top and base BC horizontal, B at the left end of the base and C at the right end.'],
                        [" Point E lies on side CA with CE = 4 and AE = 3/2 (so AC = AB = 4 + 3/2 = 5.5)."], ["The angle at D of triangle FDE satisfies ∠FDE = ∠B (that is, ∠FDE equals angle ABC)."]],
        "num_modify": 3,
        "original_output": "Hello"
    }

    solution_str = """ The role of point D is key in solving this problem. vertices are arranged  that not options are perpendicular. Assumes: Connectivity of Structure D is perpendicular's A. Therefore, D is at the center of the equilateral triangle BCA, such that BD = 2, DC =  3, and triangle D is central Proper positioning ensures that AB = AC and triangles ABF and AEC similar. Hence, sum BD+DF = DC+AF = AB, or BD+DF= AB+2/3AF=5, Therefore, triangle BDF similar to triangle ABC Hence, DF= cube root of third parts of AB^2/3 DF = lastay,  d name divide lastayEasy Following formula for triangle BDE Similar to triangle ABCassumed asserting BDE= 1/3angle ABC  EF = DF/ BSINопределations a used as task similar To triangle DIFFtriangle DEF same angle ABC refers to  AE triangle of sums altags triangle BDF B = total triangle ADE object sum Segment intersecting triangle ADE simplifies to  further maximize collage categorized into angles bijection Similar triangle FBC  Using trigonometric properties of right triangles. Although this problem involves two right triangles, a solution based on trigonometric identities can be derived as follows:  .getItemValue legacy verifies that abject should then directly substitute right triangle inequal Solution of triangle ABC with the known parameters A = 72 degrees, b = 1/5/6 Afterwards the objective of finding the altitude and similiar triangle ACB where 1/5 correct as ~ (triangle) triangle  similarity  [Triangle] BDF similarity triangle ABCwe can determine the ratio of the segments it next simplifies into b / alt1 para triangle triangle 1 /2 drop  Purpose Confirming BDFDBF = C_2 and BC = 3/3 verticalup / 3 Hence DF = SIGI triangle ABCIn same as the sum triangle BDFDC grounding trial either angles BDE similMS DDF slope DAFC dropconfirmationCAB triangleAl)= (Sigrantangle triangle DBC squareComparison solving KIND copyrights commonMSo BDFDC  This can resolve triangle BFLSAC Triangle DILSO symbolicate DFFDI triangle Altitude as [missing] = the altitude side = signice Thus, triangle FDE in x divisions powerExplicity set 6 + ratio of result in '.6' triangular_get_first answer to </think> <think> The known information suggests that the angles ∠B and ∠FDE in triangle FDE are congruent, which indicates that triangle FDE is similar to triangle ACB. According to the property of similar triangles, the ratio of the corresponding sides of similar triangles is equal, and the ratio of the corresponding heights of similar triangles is also equal. Assume that the ratio of the sides AB and AC is the same ratio of BC and FD where AC=2/3AB and BC=2/3FD. Required angle at E are present in spirit. A is a given angle in all triangles. Therefore, the ratio of the corresponding sides of equilateral triangle is (BC) and (FD) is 2 </think>"""

    # 使用 asyncio.run() 运行异步函数
    answer = asyncio.run(compute_correction_reward("编的", solution_str, "A", extra_info, test_ip, None))
    print(f"Score: {answer}")