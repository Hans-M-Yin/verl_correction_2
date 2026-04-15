from openai import OpenAI
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key="sk-Eek0pnQdHdSPIvwfYMvDoCWoI0H6QoakugWZfCViQeWJlIsD",
)

json_file_path = "rollouts_saved/test_0223_mix_data_7b_REDESIGN_REWARD/val/210.jsonl"
list_data = []
with open(json_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 跳过空行
            list_data.append(json.loads(line))

def run_item(item):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""请你判断下面针对问题的回答是不是正确的。
回答：{item['output']}
标准答案：{item['gts']}

如果正确，请你输出 "CORRECT"， 如果不正确，回答 "INCORRECT"。不要输出其他额外内容
"""
                }
            ]
        }
    ]
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-5-mini"
    )
    response = response.choices[0].message.content
    correct = True
    if "INCORRECT" in response:
        correct = False
    if (correct and item['acc_reward'] != 1) or ((not correct) and item['acc_reward'] == 1):
        return {"consistent":False, "judge":correct, "output": item['output'], "answer": item['gts']}
    else:
        return {"consistent":True, "judge":correct, "output": item['output'], "answer": item['gts']}

with Pool(processes=int(0.8 * cpu_count())) as pool:
    dataset_list = list(tqdm(pool.imap(run_item, list_data), total=len(list_data)))

consistent = []
for idx,i in enumerate(dataset_list):
    consistent.append(i['consistent'])
    if not i['consistent']:
        print(f"{i['output']}\n### {i['answer']} ### GPT judge: {i['judge']}")
print(f"{sum(consistent)} / {len(consistent)}")