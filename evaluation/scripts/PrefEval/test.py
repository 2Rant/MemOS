import json
import re

def filter_data_by_line_number(raw_data_path, processed_jsonl_path, output_path):
    """
    1. 从已处理 JSONL 文件中提取核心 ID（作为行号）。
    2. 根据这些行号，从原始 JSONL 文件中筛选掉对应的行。
    
    Args:
        raw_data_path (str): 原始 JSONL 文件的路径。
        processed_jsonl_path (str): 包含已处理数据的 JSONL 文件的路径。
        output_path (str): 存储未处理数据的新 JSONL 文件路径。
    """
    
    # 编译正则表达式：用于从 user_id 中提取数字，该数字被认为是行号
    # 假设 user_id 仍然是 "memobase_user_pref_eval_437_1027-2"，我们需要提取 437
    pattern = re.compile(r'memobase_user_pref_eval_(\d+)_1027-2')
    
    # --- 1. 读取已处理的 JSONL 文件并提取核心 ID (行号) ---
    processed_line_numbers = set()
    try:
        with open(processed_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num_in_processed, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    user_id = data.get("user_id", "")
                    
                    match = pattern.search(user_id)
                    if match:
                        # 提取到的数字（如 '437'）被视为原始文件的行号
                        line_number_str = match.group(1) 
                        processed_line_numbers.add(int(line_number_str))
                    # else: 忽略格式不符的 user_id
                        
                except json.JSONDecodeError:
                    print(f"❌ JSON 解析错误 (已处理文件)，行号: {line_num_in_processed}。已跳过此行。")
                except ValueError:
                    print(f"❌ ID 转换为数字失败 (已处理文件)，行号: {line_num_in_processed}。已跳过此行。")

        print(f"✅ 已成功从 {processed_jsonl_path} 中加载 {len(processed_line_numbers)} 个**待跳过的行号**。")
            
    except FileNotFoundError:
        print(f"❌ 错误：找不到已处理 JSONL 文件：{processed_jsonl_path}")
        return

    # --- 2. 读取原始数据并进行基于行号的筛选 ---
    unprocessed_count = 0
    processed_count = 0
    
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            # 遍历原始 JSONL 文件中的每一行，line_num 即为物理行号 (从 1 开始)
            for line_num, line in enumerate(infile, 1):
                # 检查当前行号是否在需要跳过的集合中
                if line_num not in processed_line_numbers:
                    # 行号未被标记为已处理，保留该行
                    outfile.write(line)
                    unprocessed_count += 1
                else:
                    # 行号已被标记为已处理，跳过该行
                    processed_count += 1
                    
        print("---")
        print(f"✨ 数据筛选完成！")
        print(f"📝 原始数据文件: {raw_data_path}")
        print(f"📝 已处理数据文件 (用于提取行号): {processed_jsonl_path}")
        print(f"📝 输出未处理数据文件: **{output_path}**")
        print(f"统计：")
        print(f"   保留的**未处理**记录数: **{unprocessed_count}**")
        print(f"   跳过的**已处理**记录数 (基于行号): **{processed_count}**")

    except FileNotFoundError:
        print(f"❌ 错误：找不到原始数据文件：{raw_data_path}")
    except Exception as e:
        print(f"❌ 发生致命错误：{e}")


# --- 匹配您的运行环境和路径 ---
RAW_DATA_FILE = '/mnt/afs/codes/ljl/MemOS-pipeline/evaluation/data/prefeval/pref_processed.jsonl'
PROCESSED_DATA_FILE = '/mnt/afs/codes/ljl/MemOS-pipeline/evaluation/results/prefeval/memobase_1027-2/pref_memobase_add.jsonl'
OUTPUT_FILE = '/mnt/afs/codes/ljl/MemOS-pipeline/evaluation/results/prefeval/memobase_1027-2/unprocessed_data_by_line.jsonl'

# 执行脚本
filter_data_by_line_number(RAW_DATA_FILE, PROCESSED_DATA_FILE, OUTPUT_FILE)