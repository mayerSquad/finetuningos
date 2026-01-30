from datasets import load_from_disk, Dataset
import re

# Load original dataset
dataset = load_from_disk("oasst_chat_2k")  # change path if needed

def convert(example):
    text = example["text"]

    # Split instruction & response
    instr_match = re.search(r"### Instruction:\s*(.*?)\s*### Response:", text, re.S)
    resp_match = re.search(r"### Response:\s*(.*)", text, re.S)

    if not instr_match or not resp_match:
        return {"text": None}

    instruction = instr_match.group(1).strip()
    response = resp_match.group(1).strip()

    formatted = (
        "<|user|>\n"
        f"{instruction}\n"
        "<|assistant|>\n"
        f"{response}"
    )

    return {"text": formatted}

# Apply conversion
new_dataset = dataset.map(convert, remove_columns=dataset.column_names)

# Remove broken rows (safety)
new_dataset = new_dataset.filter(lambda x: x["text"] is not None)

# Save reformatted dataset
new_dataset.save_to_disk("tinyllama_ready_data")

print("✅ Reformatted dataset saved as 'tinyllama_ready_data'")
print("Samples:", len(new_dataset))
print("\n=== SAMPLE ===\n")
print(new_dataset[0]["text"])
