from datasets import load_from_disk
import textwrap

dataset = load_from_disk("oasst_chat_2k")

print("Dataset type:", type(dataset))

# If DatasetDict, pick train
if isinstance(dataset, dict):
    print("Splits:", dataset.keys())
    dataset = dataset["train"]

print("\nColumns:", dataset.column_names)
print("\nNumber of samples:", len(dataset))

print("\n===== RAW SAMPLE (index 0) =====")
print(dataset['text'][:5])
'''
sample = dataset['text'][:5]

for k, v in sample.items():
    print(f"\n--- {k} ---")
    if isinstance(v, str):
        print(textwrap.fill(v[:1000], 100))
    else:
        print(v)'''
