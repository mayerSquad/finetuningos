import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "mistral-lora-final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("mps")

with open("eval_prompts.txt","r") as f:
    prompts=[line.strip() for line in f if line.strip()]


results = []

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7
        )
    end = time.time()

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    tokens = output.shape[-1] - inputs["input_ids"].shape[-1]

    results.append({
        "prompt": prompt,
        "response": response,
        "latency_sec": round(end - start, 3),
        "tokens_per_sec": round(tokens / (end - start), 2)
    })

with open("mistral_lora_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("📊 Benchmark saved")
