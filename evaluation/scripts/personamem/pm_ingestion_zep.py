import os
import sys
import argparse
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep
from zep_cloud import Message
import csv
import json
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
async def ingest_session(session, user_id, session_id, client):
    try:
        # 尝试创建图（如果已存在则忽略错误）
        try:
            await client.graph.create(graph_id=user_id)
        except Exception:
            pass
        
        # 添加消息到图
        for idx, msg in enumerate(session):
            try:
                # 跳过系统消息（可选）
                # if msg["role"] == "system":
                #     continue
                    
                await client.graph.add(
                    data=f"{msg['role']}: {msg['content']}",
                    type="message",
                    graph_id=user_id
                )
            except Exception as e:
                print(f"❌ Failed to add message {idx} for user {user_id}: {str(e)}")
                # 跳过此消息继续处理下一个
                continue
        
        print(f"[ZEP] ✅ Session [{session_id}]: Ingested {len(session)} messages")
        return True
    except Exception as e:
        print(f"❌ Failed to ingest session {session_id} for user {user_id}: {str(e)}")
        return False

async def main(frame, version, num_workers=2):
    print("\n" + "=" * 80)
    print(f"🚀 PERSONAMEM INGESTION - ZEP v{version}".center(80))
    print("=" * 80)

    load_dotenv()
    try:
        client = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))
        print(f"🔌 Connected to Zep API at: {os.getenv('ZEP_BASE_URL', 'default URL')}")
    except Exception as e:
        print(f"❌ Failed to initialize Zep client: {str(e)}")
        return

    question_csv_path = "data/personamem/questions_32k copy.csv"
    context_jsonl_path = "data/personamem/shared_contexts_32k.jsonl"
    total_rows = count_csv_rows(question_csv_path)

    print(f"📚 Loaded PersonaMem dataset from {question_csv_path} and {context_jsonl_path}")
    print("-" * 80)

    start_time = datetime.now()
    all_data = list(load_rows_with_context(question_csv_path, context_jsonl_path))
    
    tasks = []
    for idx, (row_data, context) in enumerate(all_data):
        user_id = f"pm_exper_user_{idx}_{version}"
        print(f"👤 User ID: {user_id}")
        
        # 尝试删除现有用户数据
        try:
            await client.user.delete(user_id)
            print(f"🗑️  Deleted existing user {user_id} from Zep memory...")
        except Exception as e:
            print(f"⚠️  Could not delete user {user_id}: {str(e)}")
        
        # 添加用户
        try:
            await client.user.add(user_id=user_id)
            print(f"➕ Added user {user_id} to Zep memory...")
        except Exception as e:
            print(f"⚠️  Could not add user {user_id}: {str(e)}")
            # 跳过此用户
            continue
        
        # 创建摄取任务
        task = ingest_session(
            session=context,
            user_id=user_id,
            session_id=idx,
            client=client
        )
        tasks.append(task)
    
    # 运行摄取任务并发
    completed = 0
    skipped = 0
    for i in range(0, len(tasks), num_workers):
        batch = tasks[i:i+num_workers]
        results = await asyncio.gather(*batch, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Batch task failed: {str(result)}")
                skipped += 1
            elif result is False:
                skipped += 1
            else:
                completed += 1
        
        print(f"✅ Batch {i//num_workers + 1}/{(len(tasks)+num_workers-1)//num_workers} completed ({completed} success, {skipped} skipped)")

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    print("\n" + "=" * 80)
    print("✅ INGESTION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time taken to ingest {total_rows} rows: {elapsed_time_str}")
    print(f"🔄 Framework: ZEP | Version: {version} | Workers: {num_workers}")
    print(f"📊 Success: {completed} | Skipped: {skipped}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaMem Ingestion Script for ZEP")
    parser.add_argument("--version", type=str, default="0925-1", help="Version of the evaluation framework.")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers for processing users.")
    args = parser.parse_args()

    asyncio.run(main(frame="zep", version=args.version, num_workers=args.workers))