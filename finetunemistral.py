import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=None
).to("mps")

# Load dataset
dataset = load_from_disk("oasst_chat_2k")

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

training_args = TrainingArguments(
    output_dir="./mistral-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    args=training_args,
)

print("🚀 Starting fine-tuning...")
trainer.train()

trainer.save_model("mistral-lora-final")
print("✅ Training complete")
