import os
import sys
import argparse
import asyncio
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep
from tqdm import tqdm
from zep_cloud.types import EntityEdge, EntityNode
import csv
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompts import ZEP_CONTEXT_TEMPLATE

def format_fact(edge: EntityEdge) -> str:
    valid_at = edge.valid_at if edge.valid_at is not None else "date unknown"
    invalid_at = edge.invalid_at if edge.invalid_at is not None else "present"
    formatted_fact = f"  - {edge.fact} (Date range: {valid_at} - {invalid_at})"
    return formatted_fact

def format_entity(node: EntityNode) -> str:
    formatted_entity = f"  - {node.name}: {node.summary}"
    return formatted_entity

def build_jsonl_index(jsonl_path):
    """
    Scan the JSONL file once to build a mapping: {key: file_offset}.
    Assumes each line is a JSON object with a single key-value pair.
    """
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

def load_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for _, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value
            yield row_data

def load_rows_with_context(csv_path, jsonl_path):
    jsonl_index = build_jsonl_index(jsonl_path)

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        prev_sid = None
        prev_context = None
        for _, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value

            sid = row_data["shared_context_id"]
            if sid != prev_sid:
                current_context = load_context_by_id(jsonl_path, jsonl_index[sid])
                prev_sid = sid
                prev_context = current_context
            else:
                current_context = prev_context
            yield row_data, current_context

def count_csv_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1

async def zep_search(client, user_id, query, top_k=20):
    try:
        # 截断过长的查询
        if len(query) > 400:
            query = query[:400]
            print(f"⚠️  Query truncated to 400 characters for user {user_id}")
        
        start = time.time()
        
        # 搜索边
        edges_results = await client.graph.search(
            query=query,
            graph_id=user_id,
            scope="edges",
            reranker="cross_encoder",
            limit=top_k
        )
        
        # 搜索节点
        nodes_results = await client.graph.search(
            query=query,
            graph_id=user_id,
            scope="nodes",
            reranker="rrf",
            limit=top_k
        )
        
        # 从结果对象中提取实际的边和节点列表
        edges_list = edges_results.edges
        nodes_list = nodes_results.nodes
        
        # 格式化结果
        facts = [format_fact(edge) for edge in edges_list]
        entities = [format_entity(node) for node in nodes_list]
        context = ZEP_CONTEXT_TEMPLATE.format(
            facts="\n".join(facts) if facts else "No facts found",
            entities="\n".join(entities) if entities else "No entities found"
        )
        
        duration_ms = (time.time() - start) * 1000
        return context, duration_ms
    except Exception as e:
        print(f"❌ Search failed for user {user_id}: {str(e)}")
        return "Search failed due to error", 0

async def process_user(row_data, conv_idx, frame, version, top_k=20):
    user_id = f"pm_exper_user_{conv_idx}_{version}"
    print(f"\n🔍 Processing conversation {conv_idx} for user {user_id}...")
    
    result_path = f"results/pm/zep-{version}/tmp/zep_pm_search_results_{conv_idx}.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    
    try:
        question = row_data["user_question_or_message"]
        persona_id = row_data["persona_id"]
        question_id = row_data["question_id"]
        question_type = row_data["question_type"]
        topic = row_data["topic"]
        correct_answer = row_data["correct_answer"]
        all_options = row_data["all_options"]
        
        load_dotenv()
        client = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))
        
        context, duration_ms = await zep_search(
            client=client,
            user_id=user_id,
            query=question,
            top_k=top_k
        )
        
        search_results = {
            user_id: [{
                "user_id": user_id,
                "question": question,
                "category": question_type,
                "persona_id": persona_id,
                "question_id": question_id,
                "all_options": all_options,
                "topic": topic,
                "golden_answer": correct_answer,
                "search_context": context,
                "search_duration_ms": duration_ms,
            }]
        }
        
        os.makedirs(f"results/pm/zep-{version}/tmp", exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(search_results, f, indent=4)
        
        print(f"💾 Search results for conversation {conv_idx} saved")
        return search_results
    except Exception as e:
        print(f"❌ Failed to process user {user_id}: {str(e)}")
        # 返回空结果表示失败
        return {user_id: [{
            "user_id": user_id,
            "error": str(e),
            "search_context": "Error occurred during processing",
            "search_duration_ms": 0
        }]}

async def main(version, top_k=20, num_workers=4):
    print("\n" + "=" * 80)
    print(f"🔍 PERSONAMEM SEARCH - ZEP v{version}".center(80))
    print("=" * 80)

    question_csv_path = "data/personamem/questions_32k copy.csv"
    total_rows = count_csv_rows(question_csv_path)

    print(f"📚 Loaded PersonaMem dataset from {question_csv_path}")
    print(f"📊 Total conversations: {total_rows}")
    print(f"⚙️  Search parameters: top_k={top_k}, workers={num_workers}")
    print("-" * 80)

    all_search_results = {}
    start_time = datetime.now()
    all_data = list(load_rows(question_csv_path))
    
    # 处理用户批次
    completed = 0
    skipped = 0
    for i in tqdm(range(0, len(all_data), num_workers), desc="Processing batches"):
        batch = all_data[i:i+num_workers]
        tasks = []
        
        for idx, row_data in enumerate(batch, start=i):
            tasks.append(
                process_user(
                    row_data=row_data,
                    conv_idx=idx,
                    frame="zep",
                    version=version,
                    top_k=top_k
                )
            )
        
        results = await asyncio.gather(*tasks)
        for result in results:
            all_search_results.update(result)
            if "error" in list(result.values())[0][0]:
                skipped += 1
            else:
                completed += 1
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    print("\n" + "=" * 80)
    print("✅ SEARCH COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time taken: {elapsed_time_str}")
    print(f"📊 Success: {completed} | Skipped: {skipped}")
    
    output_path = f"results/pm/zep-{version}/zep_pm_search_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_search_results, f, indent=4)
    
    print(f"📁 Results saved to: {output_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaMem Search Script for ZEP")
    parser.add_argument("--version", type=str, default="0925-1", help="Version of the evaluation framework.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top results to retrieve.")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers.")
    args = parser.parse_args()

    asyncio.run(main(version=args.version, top_k=args.top_k, num_workers=args.workers))