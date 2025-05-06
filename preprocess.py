# preprocess.py

from multiprocessing import freeze_support
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen1.5-4B-Chat"
SYS_PROMPT = "你是一名资深心理咨询师，请根据用户的描述给出专业而富有同理心的回复，仅输出回答内容。"

# 顶层初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

def preprocess(example):
    # 1. 构造 raw sequence
    seq = (
        "<|im_start|>system\n"     + SYS_PROMPT   + "<|im_end|>\n"
        "<|im_start|>user\n"       + example["input"]   + "<|im_end|>\n"
        "<|im_start|>assistant\n"  + example["output"]  + "<|im_end|>\n"
    )
    toks = tokenizer(
        seq,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

    # 2. 尝试直接找 "<|im_start|>assistant" 这个整体 token
    assistant_token = "<|im_start|>assistant"
    assistant_id = tokenizer.convert_tokens_to_ids(assistant_token)

    if assistant_id is not None and assistant_id in toks["input_ids"]:
        # 情况 A：整体 special token 存在
        idx = toks["input_ids"].index(assistant_id)
    else:
        # 情况 B：只把 "<|im_start|>" 当 special token
        start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        # 在 input_ids 中找到所有 start_id 的位置
        positions = [i for i, tid in enumerate(toks["input_ids"]) if tid == start_id]
        if len(positions) < 3:
            raise ValueError(f"找不到第3个 '<|im_start|>'，请检查特殊 token 配置")
        # 第三个出现的位置，就是 assistant 段的开头
        idx = positions[2]

    # 3. 构造 labels：system+user 部分全部 -100，assistant 段开始算真实标签
    labels = [-100] * idx + toks["input_ids"][idx:]
    toks["labels"] = labels[: len(toks["input_ids"])]
    return toks

def main():
    ds = load_dataset(
        "json",
        data_files={
            "train": "./prepared_data/train.jsonl",
            "valid": "./prepared_data/valid.jsonl"
        },
    )
    tokenized = ds.map(
        preprocess,
        remove_columns=ds["train"].column_names,
        # 单进程或多进程都可以
        # num_proc=1
    )
    tokenized.save_to_disk("./cache/qwen_psych")

if __name__ == "__main__":
    freeze_support()
    main()