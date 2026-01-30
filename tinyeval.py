import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# Paths
# --------------------------------------------------
MODEL_PATH = "tinyllama-lora-final"
PROMPT_FILE = "eval_prompts.txt"
OUTPUT_FILE = "tinyllama_lora_results.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --------------------------------------------------
# Load tokenizer + model
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32
).to(device)
print("model is",model)
model.eval()
print("✅ TinyLlama LoRA loaded on", device)

# --------------------------------------------------
# Load evaluation prompts
# --------------------------------------------------
with open(PROMPT_FILE, "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

print(f"📄 Loaded {len(prompts)} prompts")

# --------------------------------------------------
# Run evaluation
# --------------------------------------------------
results = []

for idx, prompt in enumerate(prompts, 1):
    formatted_prompt = (
    "<|user|>\n"
    + prompt +
    "\n<|assistant|>\n"
)


    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)


    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            min_new_tokens=20,
            temperature=0.6,
            top_p=0.85,
            

            do_sample=True,
                repetition_penalty=1.25,

            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    end = time.time()

    generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    latency = end - start
    gen_tokens=outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True
    )
    for bad in ["[INST]", "You are a helpful assistant","Responding"]:
        if bad in response:
            response=response.split(bad)[-1]
    response=response.strip()

    results.append({
        "id": idx,
        "prompt": prompt,
        "response": response,
        "latency_sec": round(latency, 3),
        "generated_tokens": generated_tokens,
        "tokens_per_sec": round(generated_tokens / latency, 2)
    })

    print(f"✔ Prompt {idx}/{len(prompts)} done")

# --------------------------------------------------
# Save results
# --------------------------------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"📊 Evaluation saved to {OUTPUT_FILE}")
