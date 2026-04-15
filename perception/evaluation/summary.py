import os
import json
import csv

def main():
    # 设定 eval_results 目录的绝对路径
    base_dir = '/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/verl/perception/evaluation/eval_results'
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return

    results = []
    
    # 遍历所有实验文件夹
    for exp_folder in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue
            
        # 提取实验名称和 step (格式形如 exp_name_global_step_128)
        if '_global_step_' in exp_folder:
            parts = exp_folder.rsplit('_global_step_', 1)
            exp_name = parts[0]
            step = parts[1]
        else:
            exp_name = exp_folder
            step = 'unknown'
            
        benchmarks = {}
        
        # 遍历文件夹内的 JSON 文件
        for filename in os.listdir(exp_path):
            if not filename.endswith('.json'):
                continue
            # 我们只处理包含了 "judge" 字样的结果文件
            if 'judge' not in filename.lower():
                continue
                
            file_path = os.path.join(exp_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 确认文件内容是列表
                if not isinstance(data, list):
                    continue
                    
                total = len(data)
                if total == 0:
                    continue
                    
                # 手动计算 Acc
                correct = 0
                for item in data:
                    judge_val = item.get('judge_result', '')
                    # 兼容布尔值与字符串的 true
                    if str(judge_val).lower() == 'true':
                        correct += 1
                        
                acc = correct / total
                
                # 从文件名判断 benchmark 名称
                lower_filename = filename.lower()
                if 'mathvista' in lower_filename:
                    b_name = 'MathVista'
                elif 'mathverse' in lower_filename:
                    b_name = 'MathVerse'
                elif 'mathvision' in lower_filename:
                    b_name = 'MathVision'
                elif 'wemath' in lower_filename:
                    b_name = 'WeMath'
                else:
                    # 对于未知的 benchmark 使用文件名作为标题
                    b_name = filename.split('.')[0]
                    
                benchmarks[b_name] = acc
            except Exception as e:
                print(f"解析文件 {file_path} 时出错: {e}")
                
        # 只要存在至少一个 benchmark 的结果，就加入到总结果列表中
        if benchmarks:
            row = {'Experiment': exp_name, 'Step': step}
            row.update(benchmarks)
            results.append(row)
            
    if not results:
        print("没有找到任何包含评测结果的 judge 文件。")
        return
        
    # 获取所有唯一的 benchmark 名称，并排序
    all_benchmarks = set()
    for row in results:
        for k in row.keys():
            if k not in ['Experiment', 'Step']:
                all_benchmarks.add(k)
    all_benchmarks = sorted(list(all_benchmarks))
    
    # 对结果进行排序 (优先按照 Experiment 名称，其次按照 Step 升序)
    def sort_key(x):
        try:
            step_val = int(x['Step'])
        except ValueError:
            step_val = -1
        return (x['Experiment'], step_val)
        
    results.sort(key=sort_key)
    
    # ---------------- 打印在终端的可视化表格 ----------------
    headers = ['Experiment', 'Step'] + all_benchmarks
    # 动态构建列宽格式
    header_format = "{:<45} | {:<8} | " + " | ".join([f"{{:<{max(len(b), 8)}}}" for b in all_benchmarks])
    
    sep_line = "-" * (45 + 8 + 3 * 2 + sum(max(len(b), 8) + 3 for b in all_benchmarks))
    print(sep_line)
    print(header_format.format(*headers))
    print(sep_line)
    
    for row in results:
        values = [row['Experiment'], row['Step']]
        for b in all_benchmarks:
            val = row.get(b, '-')
            if isinstance(val, float):
                # 以百分比显示
                values.append(f"{val:.2%}")
            else:
                values.append(val)
        print(header_format.format(*values))
        
    print(sep_line)
    
    # ---------------- 将结果保存为 CSV 文件 ----------------
    output_csv = '/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/verl/perception/evaluation/evaluation_summary.csv'
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in results:
                csv_row = {'Experiment': row['Experiment'], 'Step': row['Step']}
                for b in all_benchmarks:
                    val = row.get(b, '')
                    if isinstance(val, float):
                        # CSV 里存小数格式（保留4位小数）
                        csv_row[b] = f"{val:.4f}"
                    else:
                        csv_row[b] = val
                writer.writerow(csv_row)
        print(f"\n✅ 汇总记录表格已成功保存至: {output_csv}")
    except Exception as e:
        print(f"保存 CSV 文件时出错: {e}")

if __name__ == '__main__':
    main()