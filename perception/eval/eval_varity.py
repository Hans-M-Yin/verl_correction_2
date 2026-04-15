import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
def extract_between(text, a, b):
    start = text.find(a)
    if start == -1:
        return ""  # 如果没找到A，返回空
    start += len(a)  # 跳过A子串本身
    end = text.find(b, start)  # 从start位置开始找B
    if end == -1:
        return ""  # 如果没找到B，返回空
    return text[start:end]

metrics = {
    "score": [],
    "acc": [],
    "correction": [],
    "repeat_penalty": [],
    "format": [],
    "acc_all_correct": [],
    "acc_all_wrong": [],
    "num_type_1": []
}
for t in range(1,220):
    cnt_score, cnt_acc, cnt_correction, cnt_repeat_penalty, cnt_format = 0,0,0,0,0
    # 读取JSONL文件
    res = []
    with open(f"rollout_new_reward/train/{str(t)}.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                res.append(json.loads(line))

    stat = defaultdict(list)
    for i in res:
        question = extract_between(i['input'], "\n\nuser\n\n", "\nassistant\n")
        if not i['input'].endswith('\nassistant\n'):
            stat[question].append(i)
        cnt_acc += i['acc_reward']
        cnt_correction += i['correction_reward']
        cnt_repeat_penalty += i['repeat_penalty']
        cnt_format += i['format_reward']
        cnt_score += i['score']
    acc_all_correct = 0
    acc_all_wrong = 0
    score_same_count = 0
    acc_all = 0
    for k, v in stat.items():
        num = len(v)
        acc = 0
        correction_reward = 0
        score = 0
        score_list = []
        for j in v:
            if j['score'] == 1.2:
                print(j['input'])
                print(j['output'])
                print("##########")
            acc += j['acc_reward']
            correction_reward += j['correction_reward']
            score += j['score']
            score_list.append(j['score'])
        print(f" ## Average accuracy:{acc / num} Correction: {correction_reward / num} Score: {score / num} Score List: {score_list}")
        if acc == num:
            acc_all_correct += 1
        elif acc == 0:
            acc_all_wrong += 1
        acc_all += acc
    metrics['score'].append(cnt_score / len(res))
    metrics['acc'].append(cnt_acc / len(res))
    metrics['correction'].append(cnt_correction / len(res))
    metrics['repeat_penalty'].append(cnt_repeat_penalty / len(res))
    metrics['format'].append(cnt_format / len(res))
    metrics['acc_all_correct'].append(acc_all_correct / len(stat))
    metrics['acc_all_wrong'].append(acc_all_wrong / len(stat))
    metrics['num_type_1'].append(len(stat))

# 可视化
plt.figure(figsize=(20, 16))
x = list(range(1, 220))  # t从1到219

# 定义子图布局：4行2列，共8个子图
rows, cols = 4, 2
metrics_order = [
    'score', 'acc', 'correction', 'repeat_penalty',
    'format', 'acc_all_correct', 'acc_all_wrong', 'num_type_1'
]
titles = {
    'score': 'Average Score',
    'acc': 'Accuracy Reward',
    'correction': 'Correction Reward',
    'repeat_penalty': 'Repeat Penalty',
    'format': 'Format Reward',
    'acc_all_correct': 'Proportion of All-Correct Questions',
    'acc_all_wrong': 'Proportion of All-Wrong Questions',
    'num_type_1': 'Number of Unique Questions'
}

for idx, key in enumerate(metrics_order, 1):
    plt.subplot(rows, cols, idx)
    plt.plot(x, metrics[key], marker='.', markersize=3, linewidth=1)
    plt.title(titles[key])
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    # 设置y轴范围，排除异常值
    y_data = metrics[key]
    if len(y_data) > 0:
        plt.ylim(min(y_data) * 0.95, max(y_data) * 1.05)

plt.tight_layout()
plt.show()


