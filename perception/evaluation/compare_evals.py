import json
import os

def load_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item['judge_result'] == 'true' for item in data}

def compare_results(path210, path280):
    res210 = load_results(path210)
    res280 = load_results(path280)
    
    all_ids = set(res210.keys()).union(set(res280.keys()))
    
    improved = []
    degraded = []
    stayed_true = []
    stayed_false = []
    missing_in_210 = []
    missing_in_280 = []
    
    for rid in all_ids:
        v210 = res210.get(rid)
        v280 = res280.get(rid)
        
        if v210 is None:
            missing_in_210.append(rid)
            continue
        if v280 is None:
            missing_in_280.append(rid)
            continue
            
        if not v210 and v280:
            improved.append(rid)
        elif v210 and not v280:
            degraded.append(rid)
        elif v210 and v280:
            stayed_true.append(rid)
        else:
            stayed_false.append(rid)
            
    print(f"Total processed IDs: {len(all_ids)}")
    print(f"Accuracy 210: {sum(res210.values())/len(res210):.2%} ({sum(res210.values())}/{len(res210)})")
    print(f"Accuracy 280: {sum(res280.values())/len(res280):.2%} ({sum(res280.values())}/{len(res280)})")
    print("-" * 30)
    print(f"Improved: {len(improved)}")
    print(f"Degraded: {len(degraded)}")
    print(f"Stayed True: {len(stayed_true)}")
    print(f"Stayed False: {len(stayed_false)}")
    
    if missing_in_210:
        print(f"Missing in 210: {len(missing_in_210)}")
    if missing_in_280:
        print(f"Missing in 280: {len(missing_in_280)}")
        
    print("-" * 30)
    print("Degraded IDs:")
    print(", ".join(sorted(degraded, key=lambda x: int(x) if x.isdigit() else x)))
    
    # Save degraded IDs to a file
    with open('degraded_ids.json', 'w') as f:
        json.dump(sorted(degraded, key=lambda x: int(x) if x.isdigit() else x), f, indent=4)
        
if __name__ == "__main__":
    path210 = r"c:\Users\13943\Downloads\a1\perception\verl\perception\evaluation\eval_0307_NO_CORRECTION_REWARD_210\mathvista.json-qwen_judge_results_v4.json"
    path280 = r"c:\Users\13943\Downloads\a1\perception\verl\perception\evaluation\eval_0307_NO_CORRECTION_REWARD_280\mathvista.json-qwen_judge_results_v4.json"
    compare_results(path210, path280)
