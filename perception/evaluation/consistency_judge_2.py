import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--original_path', type=str, help='original consistency judge path')
parser.add_argument('--consistency_judge_path', type=str, help='consistency judge path')
args = parser.parse_args()
original_output_path = args.original_path
consistency_judge_path = args.consistency_judge_path

with open(original_output_path, 'r', encoding='utf-8') as f:
    original_path = json.load(f)

with open(consistency_judge_path, 'r', encoding='utf-8') as f:
    consistency_judge = json.load(f)

correct_in_inconsistent = 0
inconsistent = 0
for i,j in zip(original_path, consistency_judge):
    if i['id'] == j['id']:
        if j['consistency'] == 'inconsistent':
            inconsistent += 1
            if i['judge_result'] == 'true':
                correct_in_inconsistent += 1

print(inconsistent)
print(correct_in_inconsistent)
print(correct_in_inconsistent/inconsistent)
