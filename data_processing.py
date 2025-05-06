import os
import json
from tqdm import tqdm
from datasets import Dataset

def process_single_dialogue(dialogue_path):
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        session = json.load(f)

    history = []
    samples = []
    
    for i in range(0, len(session)-1, 2):
        client_msg = session[i]
        counselor_msg = session[i+1] if i+1 < len(session) else None

        if client_msg["role"] != "client" or not counselor_msg or counselor_msg["role"] != "counselor":
            continue

        user_text = client_msg["content"].strip()
        assistant_text = counselor_msg["content"].strip()

        # 构造prompt（包括历史）
        prompt = ""
        for past_user, past_assistant in history:
            prompt += f"<|user|>\n{past_user}\n<|assistant|>\n{past_assistant}\n"
        prompt += f"<|user|>\n{user_text}\n<|assistant|>\n"

        samples.append({
            "prompt": prompt,
            "response": assistant_text
        })

        # 更新历史
        history.append((user_text, assistant_text))

    return samples

def build_dataset_from_folder(folder_path):
    all_samples = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    for file in tqdm(files, desc="Processing JSON files"):
        full_path = os.path.join(folder_path, file)
        all_samples.extend(process_single_dialogue(full_path))
    return Dataset.from_list(all_samples)

# 用法示例
dataset = build_dataset_from_folder("./data")
dataset.save_to_disk("./mechat_sft_dataset")
