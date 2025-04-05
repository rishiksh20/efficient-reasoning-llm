# verifier_dataset.py
import json
import re
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = "You are a math expert who explains solutions step by step."
QUESTION_TEMPLATE = "\n### Question:\n{question}\n\n### Answer:\n"

def format_prompt(example):
    return SYSTEM_PROMPT + QUESTION_TEMPLATE.format(question=example["question"].strip())

def extract_answer(text):
    match = re.search(r"The answer is (.*?)(\.|$)", text)
    return match.group(1).strip() if match else None

def build_verifier_dataset(model_outputs_file, output_jsonl="verifier_data.jsonl"):
    dataset = load_dataset("gsm8k", "main", split="test")
    gold_lookup = {
        ex["question"].strip(): ex["answer"].split("####")[-1].strip()
        for ex in dataset
    }

    with open(model_outputs_file, "r") as f:
        generations = json.load(f)  # Expecting {question: [list of generations]}

    output_lines = []
    for question, samples in generations.items():
        gold = gold_lookup.get(question.strip())
        for gen in samples:
            answer = extract_answer(gen)
            label = int(answer == gold)
            prompt = format_prompt({"question": question})
            full = prompt + gen
            output_lines.append({"text": full, "label": label})

    with open(output_jsonl, "w") as f:
        for ex in output_lines:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved verifier dataset with {len(output_lines)} examples to {output_jsonl}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True, help="Path to generated outputs JSON")
    args = parser.parse_args()
    build_verifier_dataset(args.gen_file)
