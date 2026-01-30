'''import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer
print("new wala ")
# --------------------------------------------------
# 1. Model
# --------------------------------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------
# 3. Model (MPS-safe)
# --------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map={"": "mps"}
)

print("✅ Model loaded on MPS")

# --------------------------------------------------
# 4. Dataset
# --------------------------------------------------
dataset = load_from_disk("tinyllama_ready_data")

# Simple instruction formatting
def format_example(example):
    return {
        "text": f"<s>[INST] {example['text']} [/INST]</s>"
    }

dataset = dataset.map(format_example)
dataset = dataset.shuffle(seed=42)

print("✅ Dataset formatted")

# --------------------------------------------------
# 5. LoRA config
# --------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# --------------------------------------------------
# 6. Training args (Mac SAFE)
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=200,
    fp16=False,
    bf16=False,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# --------------------------------------------------
# 7. Trainer
# --------------------------------------------------
def fun(example):
    return example["text"]
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
    args=training_args,
    formatting_func=fun,
    
)

# --------------------------------------------------
# 8. Train
# --------------------------------------------------
print("🚀 Starting TinyLLaMA LoRA fine-tuning...")
print(dataset[0]["text"])

trainer.train()

# --------------------------------------------------
# 9. Save adapter
# --------------------------------------------------
trainer.save_model("tinyllama-lora-final")
print("✅ Training complete")'''
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

print("🚀 TinyLLaMA LoRA Training (JSONL)")

# --------------------------------------------------
# 1. Model
# --------------------------------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------
# 3. Model (MPS-safe for Mac)
# --------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map={"": "mps"}
)

print("✅ Model loaded on MPS")

# --------------------------------------------------
# 4. Dataset (JSONL)
# --------------------------------------------------
dataset = load_dataset(
    "json",
    data_files="alpaca_tinyllama.jsonl",
    split="train"
)

dataset = dataset.shuffle(seed=42)

print("✅ Dataset loaded")
print(dataset[0]["text"])

# --------------------------------------------------
# 5. LoRA config
# --------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# --------------------------------------------------
# 6. Training args (Mac SAFE)
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=200,
    fp16=False,
    bf16=False,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# --------------------------------------------------
# 7. Trainer
# --------------------------------------------------
def fun(example):
    return example["text"]
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
    args=training_args,
    formatting_func=fun,
    
)

# --------------------------------------------------
# 8. Train
# --------------------------------------------------
print("🔥 Starting TinyLLaMA LoRA fine-tuning...")
trainer.train()

# --------------------------------------------------
# 9. Save adapter
# --------------------------------------------------
trainer.save_model("tinyllama-lora-final")
print("✅ Training complete")
