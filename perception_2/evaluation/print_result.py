import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_files', type=str)
args = parser.parse_args()
with open(args.json_files,'r', encoding='utf-8') as f:
    results = json.load(f)

correct = 0
for i in range(len(results)):
    correct += results[i]['judge_result'] == 'true'

print(correct / len(results))
