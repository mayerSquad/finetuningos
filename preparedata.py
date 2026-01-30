from datasets import load_dataset

# Load subset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")
train_ds= dataset
print(len(train_ds))
def alpaca_to_tinyllama(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip():
        user_content = instruction + "\n" + input_text
    else:
        user_content = instruction

    text = (
        "<|user|>\n"
        + user_content
        + "\n<|assistant|>\n"
        + output
        + "\n<|endoftext|>"
    )

    return {"text": text}
tinyllama_dataset = train_ds.map(
    alpaca_to_tinyllama,
    remove_columns=train_ds.column_names
)
print(tinyllama_dataset[0]["text"])
tinyllama_dataset.to_json("alpaca_tinyllama.jsonl")

'''
# Step 1: filter only assistant messages
dataset = dataset.filter(lambda x: x["role"] == "assistant")

# Step 2: format
def format_example(example):
    prompt = f"""### Instruction:
Respond to the user

### Response:
{example['text']}"""
    return {"text": prompt}

dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names
)

dataset.save_to_disk("oasst_chat_2k")
print("✅ Dataset prepared:", len(dataset))'''
