import json
import os
import sys
import time
import traceback
import argparse
import csv
from datetime import datetime
from tqdm import tqdm
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mirix_utils import get_mirix_client

# 辅助函数：加载上下文数据
def build_jsonl_index(jsonl_path):
    index = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index

def load_context_by_id(jsonl_path, offset):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))

def load_rows_with_context(csv_path, jsonl_path):
    jsonl_index = build_jsonl_index(jsonl_path)
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        prev_sid = None
        prev_context = None
        for row in reader:
            sid = row["shared_context_id"]
            if sid != prev_sid:
                current_context = load_context_by_id(jsonl_path, jsonl_index[sid])
                prev_sid = sid
                prev_context = current_context
            else:
                current_context = prev_context
            yield row, current_context

def count_csv_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1

# 答案匹配函数
import re

def is_answer_correct(predicted, correct, options=None):
    """检查预测答案是否正确，支持带括号的正确答案格式"""
    # 规范化字符串：去除首尾空格/括号并转小写
    def normalize(s):
        s = s.strip().lower()
        # 移除首尾括号（如果存在）
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1].strip()
        return s

    # 1. 直接比较规范化后的字符串
    norm_pred = normalize(predicted)
    norm_corr = normalize(correct)
    
    if norm_pred == norm_corr:
        return True
    
    # 2. 尝试从预测答案中提取选项字母
    # 匹配格式如 "(c) ...", "c) ...", "[c] ..." 等
    letter_match = re.match(r'^\s*[\[(]?\s*([a-zA-Z])\s*[)\]]?\s*', predicted)
    if letter_match:
        extracted_letter = letter_match.group(1).lower()
        if extracted_letter == norm_corr:
            return True
    
    # 3. 处理选项匹配
    if options:
        # 解析选项格式 (如 "A. Paris B. London C. Berlin")
        option_map = {}
        # 使用正则分割选项，支持 A. 和 A) 两种格式
        parts = re.split(r'([A-Z][.)]\s*)', options)
        parts = [p.strip() for p in parts if p.strip()]
        
        # 构建选项映射 {字母: 内容}
        for i in range(0, len(parts), 2):
            if i+1 < len(parts):
                # 提取选项字母 (A. -> A, A) -> A)
                letter = parts[i][0].lower()
                content = parts[i+1].strip().lower()
                option_map[letter] = content

        # 检查两种情况：
        # a) 预测答案匹配正确选项内容
        if norm_corr in option_map:
            correct_content = option_map[norm_corr]
            
            # 比较预测答案和选项内容（忽略选项字母）
            if norm_pred == correct_content:
                return True
            
            # 比较预测答案和选项字母
            if norm_pred == norm_corr:
                return True
        
        # b) 预测答案包含正确选项内容
        for letter, content in option_map.items():
            # 如果预测答案包含正确选项内容
            if content in norm_pred:
                # 并且字母匹配
                if letter == norm_corr:
                    return True
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaMem Evaluation with Mirix")
    parser.add_argument("--version", type=str, default="v1", help="Evaluation version")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    config_path = "./scripts/personamem/config.yaml"
    question_csv_path = "data/personamem/questions_32k.csv"
    context_jsonl_path = "data/personamem/shared_contexts_32k.jsonl"
    
    # 创建结果目录
    results_dir = f"./results/pm/mirix-{args.version}"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, "mirix_pm_test_result.json")
    summary_file = os.path.join(results_dir, "mirix_pm_responses.json")
    stats_file = os.path.join(results_dir, "mirix_pm_accuracy_stats.json")
    
    success_results = []
    success_qid = []

    # 加载已有结果
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            for line in f:
                try:
                    res = json.loads(line)
                    if res.get("response") and "ERROR" not in res["response"]:
                        success_results.append(res)
                        success_qid.append(res["question_id"])
                except:
                    continue
    
    # 初始化Mirix客户端
    assistant = get_mirix_client(config_path)
    
    # 处理单个问题
    def process_one(row_data, context):
        end_index = int(row_data["end_index_in_shared_context"])
        truncated_context = context[:end_index]
        q = row_data["user_question_or_message"]
        a = row_data["correct_answer"]
        q_type = row_data["question_type"]
        question_id = row_data["question_id"]
        persona_id = row_data["persona_id"]
        all_options = row_data["all_options"]
        topic = row_data["topic"]
        
        # 创建用户
        user_id = f"pm_user_{persona_id}_{question_id}"
        assistant.create_user(user_name=user_id)
        user = assistant.get_user_by_name(user_name=user_id)
        
        # 添加记忆
        message = f"Conversation:\n\n"
        for turn in truncated_context:
            message += turn["role"] + ": " + turn["content"] + "\n"
        assistant.add(message, user_id=user.id)
        
        # 设置核心记忆
        assistant._agent.update_core_memory_persona(
            "Is a helpful assistant that answers questions with extreme conciseness."
        )
        
        # 提问并计时
        start_time = time.time()
        response = assistant.chat(
            f"""Question: {q}\n\nOptions:{all_options}\n\nPlease output only with the letter of the options , such as a,b,c and d """,
            user_id=user.id,
        )
        response_duration_ms = (time.time() - start_time) * 1000
        
        # 检查答案是否正确
        is_correct = is_answer_correct(response, a, all_options)
        
        # 构建结果
        res = {
            "question_id": question_id,
            "persona_id": persona_id,
            "question": q,
            "golden_answer": a,
            "response": response,
            "is_correct": is_correct,
            "question_type": q_type,
            "response_duration_ms": response_duration_ms,
            "all_options": all_options,
            "topic": topic
        }
        return res

    # 主处理循环
    total_rows = count_csv_rows(question_csv_path)
    processed = 0
    correct_count = 0
    
    with open(result_file, "a+") as f:
        for row_data, context in tqdm(
            load_rows_with_context(question_csv_path, context_jsonl_path),
            total=total_rows,
            desc="Processing PersonaMem"
        ):
            question_id = row_data["question_id"]
            if question_id in success_qid:
                continue
                
            try:
                res = process_one(row_data, context)
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                processed += 1
                
                # 更新正确计数
                if res["is_correct"]:
                    correct_count += 1
            except Exception as exc:
                traceback.print_exc()
                print(f"❌ Error processing {question_id}: {exc}")
    
    # 汇总结果
    final_results = {}
    category_stats = {}
    topic_stats = {}
    
    # 加载所有结果（包括之前处理的和新处理的）
    with open(result_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                qid = data["question_id"]
                final_results[qid] = {
                    "question": data["question"],
                    "golden_answer": data["golden_answer"],
                    "answer": data["response"],
                    "is_correct": data["is_correct"],
                    "category": data["question_type"],
                    "response_duration_ms": data["response_duration_ms"],
                    "all_options": data["all_options"],
                    "topic": data["topic"]
                }
                
                # 更新类别统计
                category = data["question_type"]
                category_stats.setdefault(category, {"total": 0, "correct": 0})
                category_stats[category]["total"] += 1
                if data["is_correct"]:
                    category_stats[category]["correct"] += 1
                
                # 更新主题统计
                topic = data["topic"]
                topic_stats.setdefault(topic, {"total": 0, "correct": 0})
                topic_stats[topic]["total"] += 1
                if data["is_correct"]:
                    topic_stats[topic]["correct"] += 1
            except:
                continue
    
    # 计算总体准确率
    total_questions = len(final_results)
    overall_accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    # 计算各类别准确率
    for category, stats in category_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    # 计算各主题准确率
    for topic, stats in topic_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    # 保存结果
    with open(summary_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    # 保存统计信息
    stats_data = {
        "overall_accuracy": overall_accuracy,
        "total_questions": total_questions,
        "correct_count": correct_count,
        "category_stats": category_stats,
        "topic_stats": topic_stats
    }
    
    with open(stats_file, "w") as f:
        json.dump(stats_data, f, indent=2)
    
    # 打印统计结果
    print("\n" + "="*80)
    print(f"📊 Evaluation Results Summary".center(80))
    print("="*80)
    print(f"✅ Overall Accuracy: {overall_accuracy:.2%} ({correct_count}/{total_questions})")
    print(f"⏱️  Average Response Time: {sum(r['response_duration_ms'] for r in final_results.values())/total_questions:.2f} ms")
    print("\n📈 Accuracy by Question Type:")
    for category, stats in category_stats.items():
        print(f"  - {category}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\n📈 Accuracy by Topic:")
    for topic, stats in topic_stats.items():
        print(f"  - {topic}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print("="*80)
    
    print(f"\n✅ Processing completed! Processed {processed} new questions.")
    print(f"Results saved to: {summary_file}")
    print(f"Statistics saved to: {stats_file}")