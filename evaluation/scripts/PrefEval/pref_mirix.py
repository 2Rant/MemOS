import os
import json
import time
import yaml
from tqdm import tqdm
import sqlite3
import json
def load_turns_data():
    with open(f"./data/prefeval/filtered_inter_turns.json", "r") as infile:
        return json.load(infile)

turns_data = load_turns_data()


def extract_multi_turn_message(turns_data, inter_turns, model_type):
    if inter_turns > 0:
        multi_turn_message = []
        for turn_data in turns_data:
            multi_turn_message.extend(turn_data["conversation"])
        return (
            extract_multi_turn_conversation(multi_turn_message, inter_turns, model_type=model_type),
            multi_turn_message,
        )
    else:
        multi_turn_message = None
    return "", multi_turn_message

def extract_multi_turn_conversation(multi_turn_message, turn_number=3, model_type="llama"):
    message = []
    for turn in multi_turn_message:
        role = turn["role"]
        content = turn["content"]
        if model_type == "llama":
            message.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type == "claude":
            message.append({"role": role, "content": content})
        elif model_type == "mistral":
            if role == "user":
                message.append(f"[INST] {content} [/INST]")
            else:
                message.append(f"{content}</s>")
        elif model_type == "gpt":
            message.append({"role": role, "content": content})
        elif model_type == "gemini":
            gemini_role = {"user": "user", "assistant": "model"}.get(role, "user")
            message.append({"role": gemini_role, "parts": [{"text": str(content)}]})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        if len(message) == turn_number * 2:
            if role != "assistant":
                raise ValueError("The last turn must be from assistant")
            break
    assert len(message) == turn_number * 2, "The number of turns is less than the specified number"
    if "llama" in model_type or "mistral" in model_type:
        message = "".join(message)
    return message
def get_mirix_client(config_path):
    """初始化并返回Mirix客户端，增加重试机制"""
    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(os.path.expanduser("~/.mirix")):
                os.system("rm -rf ~/.mirix/*")
            
            with open(config_path, "r") as f:
                agent_config = yaml.safe_load(f)
            
            os.environ['OPENAI_API_KEY'] = agent_config['api_key']
            import mirix
            from mirix import Mirix, EmbeddingConfig, LLMConfig
            
            # 创建默认配置
            embedding_config = EmbeddingConfig(
                embedding_model=agent_config['embedding_model_name'],
                embedding_endpoint_type="openai",
                embedding_endpoint=agent_config['model_endpoint'],
                embedding_dim=1536,
                embedding_chunk_size=8191,
            )
            
            llm_config = LLMConfig(
                model=agent_config['model_name'],
                model_endpoint_type="openai",
                model_endpoint=agent_config['model_endpoint'],
                api_key=agent_config['api_key'],
                model_wrapper=None,
                context_window=128000,
            )
            
            # 设置默认配置函数
            def embedding_default_config(cls, model_name=None, provider=None):
                return embedding_config
            
            def llm_default_config(cls, model_name=None, provider=None):
                return llm_config
            
            mirix.EmbeddingConfig.default_config = embedding_default_config
            mirix.LLMConfig.default_config = llm_default_config
            
            # 创建并返回Mirix助手
            return Mirix(
                api_key=agent_config['api_key'],
                config_path=config_path,
                model=agent_config['model_name']
            )
        
        except (sqlite3.OperationalError, ImportError, Exception) as e:
            if "no such table" in str(e) or attempt == max_retries - 1:
                print(f"Mirix初始化失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    # 强制清理Mirix目录
                    if os.path.exists(os.path.expanduser("~/.mirix")):
                        os.system("rm -rf ~/.mirix/*")
                else:
                    raise Exception(f"Mirix初始化失败，已达最大重试次数: {e}")
            else:
                raise e

def process_line(line, config_path, index):
    """处理单行数据，增加重试机制"""
    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            data = json.loads(line)
            conversation = data.get("conversation", [])
            question = data.get("question")
            
            if not question:
                data["response"] = "Question not found"
                return data
            
            # 初始化Mirix客户端
            assistant = get_mirix_client(config_path)
            multi_inter_message, _ = extract_multi_turn_message(turns_data, 10, "gpt")
            # 将对话转换为字符串格式
            conv_str = "\n".join([f"{msg['role']}:{msg['content']}" for msg in conversation+multi_inter_message])
            
            # 添加对话到Mirix
            start_add = time.monotonic()
            if conv_str:
                assistant.add(conv_str)
            add_duration = time.monotonic() - start_add
            
            # 等待Mirix处理数据
            time.sleep(10)
            
            # 提问并获取回答
            start_chat = time.monotonic()
            response = assistant.chat(question)
            chat_duration = time.monotonic() - start_chat
            
            # 更新数据
            data["response"] = response
            data["metrics"] = {
                "add_duration_seconds": add_duration,
                "chat_duration_seconds": chat_duration,
                "attempts": attempt + 1  # 记录尝试次数
            }
            time.sleep(2)
            return data
        
        except (sqlite3.OperationalError, Exception) as e:
            if "no such table" in str(e):
                print(f"数据库错误处理行 {index} (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    # 强制清理Mirix目录
                    if os.path.exists(os.path.expanduser("~/.mirix")):
                        os.system("rm -rf ~/.mirix/*")
                else:
                    # 最后一次尝试也失败，返回错误信息
                    data = json.loads(line)
                    data["response"] = f"Error after {max_retries} attempts: {str(e)}"
                    data["metrics"] = {
                        "add_duration_seconds": 0,
                        "chat_duration_seconds": 0,
                        "error": True,
                        "attempts": attempt + 1
                    }
                    return data
            else:
                # 非数据库错误，直接返回
                data = json.loads(line)
                data["response"] = f"Error: {str(e)}"
                data["metrics"] = {
                    "add_duration_seconds": 0,
                    "chat_duration_seconds": 0,
                    "error": True,
                    "attempts": attempt + 1
                }
                return data

def get_processed_indices(output_file):
    """获取已处理的行索引"""
    processed_indices = set()
    if not os.path.exists(output_file):
        return processed_indices
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 检查是否有响应字段，如果有则认为已处理
                    if "response" in data:
                        # 尝试从行内容中提取索引信息（如果有的话）
                        # 这里我们简单地将所有行视为已处理
                        processed_indices.add(len(processed_indices))
                except:
                    continue
    except Exception as e:
        print(f"Error reading output file: {e}")
    
    return processed_indices

def main():
    # 配置文件路径
    config_path = './scripts/personamem/config.yaml'
    
    # 输入输出文件路径
    input_file = "./data/prefeval/pref_processed.jsonl"
    output_file = "./results/mirix/pref_mirix_1000.jsonl"
    
    print(f"Starting processing for file: {input_file}")
    
    # 获取已处理的行索引
    processed_indices = get_processed_indices(output_file)
    print(f"Found {len(processed_indices)} already processed lines")
    
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found '{input_file}'")
        return
    
    # 处理未处理的行
    processed_count = 0
    with open(output_file, 'a', encoding='utf-8') as outfile:  # 使用追加模式
        for i, line in enumerate(tqdm(lines, desc="Processing lines")):
            # 跳过已处理的行
            if i in processed_indices:
                continue
                
            result = process_line(line, config_path, i)
            if result:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                outfile.flush()  # 确保每条结果都立即写入文件
                processed_count += 1
                
                # 每处理5行后暂停一下，减少数据库压力
                if processed_count % 5 == 0:
                    time.sleep(3)
    
    print(f"\nProcessing complete! Successfully processed {processed_count} new lines.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()