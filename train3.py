# train_fp16_lora_qwen.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk

MODEL_NAME = "Qwen1.5-4B-Chat"
CACHE_PATH = "./cache/qwen_psych"

# 1) 载入 FP16 模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    # torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.gradient_checkpointing_enable()

model.config.use_cache = False
# 1.1) 单独加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# 2) LoRA 配置（同之前）
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_cfg)

# 3) 加载数据
datasets = load_from_disk(CACHE_PATH)

# 4) 训练参数
training_args = TrainingArguments(
    output_dir="outputs/qwen_fp16_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    # fp16=True,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# 5) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    tokenizer=tokenizer,
)

# 6) 训练
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./qwen4b-psychology-lora")
    tokenizer.save_pretrained("./qwen4b-psychology-lora")
