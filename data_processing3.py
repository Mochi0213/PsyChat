import glob
import json
import random
from pathlib import Path
from tqdm import tqdm

# 1. 配置
DATA_DIR    = Path("./data")
OUTPUT_DIR  = Path("./prepared_data")
OUTPUT_DIR.mkdir(exist_ok=True)
INSTRUCTION = (
    "你是一名资深心理咨询师，"
    "请根据用户的描述给出专业而富有同理心的回复，仅输出回答内容。"
)

# 2. 从一条对话list中抽取 client→counselor 样本
def extract_pairs(dialogs):
    samples = []
    for i in range(len(dialogs) - 1):
        if dialogs[i]["role"] == "client" and dialogs[i+1]["role"] == "counselor":
            user   = dialogs[i]["content"].strip()
            answer = dialogs[i+1]["content"].strip()
            samples.append({
                "instruction": INSTRUCTION,
                "input":       user,
                "output":      answer
            })
    return samples

# 3. 遍历所有 JSON 文件，累积样本
all_samples = []
for path in tqdm(DATA_DIR.glob("*.json"), total=len(list(DATA_DIR.glob("*.json")))):
    dialogs = json.loads(path.read_text(encoding="utf-8"))
    pairs   = extract_pairs(dialogs)
    all_samples.extend(pairs)

print(f"共提取到 {len(all_samples)} 条训练样本。")

# 4. 随机打乱并拆分
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.9)
train_samples = all_samples[:split_idx]
valid_samples = all_samples[split_idx:]

# 5. 写入 JSONL
for name, samples in [("train.jsonl", train_samples), ("valid.jsonl", valid_samples)]:
    out_path = OUTPUT_DIR / name
    with out_path.open("w", encoding="utf-8") as fout:
        for s in samples:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"已写入 {out_path} ({len(samples)} 条)")

