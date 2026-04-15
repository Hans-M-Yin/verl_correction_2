import json

output_file = "rollout_saved/90.jsonl"

output = []
with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 跳过空行
            output.append(json.loads(line))
keywords = ["however", "wait", "re-evaluate", "but", "although",
            "nevertheless", "on the other hand", "actually", "re-examine"]

count = 0
correct_count = 0
for i in output:
    if i['acc_reward'] > 0.5:
        correct_count += 1
    # if i['repeat_penalty'] < 0.8 and "\nassistant\n<think>" not in i['input']:
        # 使用any()和生成器表达式
    if any(keyword in i['output'].lower() for keyword in keywords):
        count += 1
        print("############################################")
        print(i['output'], i['reward'], i['acc_reward'], i['correction_reward'], i['repeat_penalty'], i['format_reward'])

print("### All:", count / len(output))
print("### Correct:", correct_count / len(output))
