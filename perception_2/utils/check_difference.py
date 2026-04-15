"""
0226，检查0224跑的实验的step_210和step_249之间的验证机差异
"""

import json

jsonl_1 = "./rollouts_saved/test_0223_mix_data_7b_REDESIGN_REWARD/val/210.jsonl"
jsonl_2 = "./rollouts_saved/test_0223_mix_data_7b_REDESIGN_REWARD/val/249.jsonl"
# jsonl_2 = ""
list_data_1 = []
with open(jsonl_1, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 跳过空行
            list_data_1.append(json.loads(line))

list_data_2 = []

with open(jsonl_2, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 跳过空行
            list_data_2.append(json.loads(line))

count_1 = 0
count_2 = 0
for l1,l2 in zip(list_data_1, list_data_2):
    if l1['acc_reward'] != l2['acc_reward']:
        if l1['acc_reward'] > l2['acc_reward']:
            count_1 += 1
            print(f"""
#############
L1 > L2: 
{l1['output']}
---------------------
{l2['output']}
---------------------
ANSWERS: {l1['gts']}
##########
            """)
        else:
            count_2 += 1

            print(f"""
#############
L1 < L2: 
{l1['output']}
---------------------
{l2['output']}
---------------------
ANSWERS: {l1['gts']}
##########
            """)

print(count_1, count_2)