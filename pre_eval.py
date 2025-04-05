# eval_gsm8k.py
import torch
import re
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lora import inject_lora_layers
from merge_lora import merge_lora_weights_into_base
from gptq_quantizer import QuantizedLinear, quantize_model_weights
from torch.utils.data import DataLoader

SYSTEM_PROMPT = "You are a math expert who explains solutions step by step."
QUESTION_TEMPLATE = "\n### Question:\n{question}\n\n### Answer:\n"

def extract_answer(text):
    match = re.search(r"The answer is (.*?)(\\.|$)", text)
    return match.group(1).strip() if match else None

def format_prompt(question):
    return SYSTEM_PROMPT + QUESTION_TEMPLATE.format(question=question.strip())

def run_eval(
    model_name_or_path,
    lora_ckpt_path,
    split="test",
    max_new_tokens=256,
    num_samples=5,
    top_p=0.95,
    temperature=0.7,
    max_length=512,
    batch_size=4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    from datasets import load_dataset as hf_load_dataset
    train_data = hf_load_dataset("gsm8k", "main", split="train[:5%]")
    from torch.utils.data import Dataset

    class PromptOnlyDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return {
                "input_ids": tokenizer(
                    SYSTEM_PROMPT + QUESTION_TEMPLATE.format(question=self.examples[idx]["question"].strip()),
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )["input_ids"].squeeze(0)
            }

    calibration_loader = DataLoader(PromptOnlyDataset(train_data), batch_size=2, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = quantize_model_weights(model, calibration_loader, num_batches=10).to(device).half()
    model = inject_lora_layers(model)
    model.load_state_dict(torch.load(lora_ckpt_path), strict=False)
    model = merge_lora_weights_into_base(model)
    model.eval()

    raw_dataset = load_dataset("gsm8k", "main", split=split)
    questions = [ex["question"] for ex in raw_dataset]
    answers = [ex["answer"].split("####")[-1].strip() for ex in raw_dataset]
    loader = DataLoader(list(zip(questions, answers)), batch_size=batch_size)

    correct = 0
    total = 0
    all_generations = {}

    with torch.inference_mode():
        for batch in tqdm(loader):
            q_batch, gold_batch = zip(*batch)
            prompts = [format_prompt(q) for q in q_batch]
            input = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

            batch_outputs = []
            for _ in range(num_samples):
                try:
                    outputs = model.generate(
                        input_ids=input["input_ids"],  # safer generation
                        attention_mask=input["attention_mask"],
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        bad_words_ids=None
                    )
                    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    batch_outputs.append(decoded)
                except RuntimeError as e:
                    print("[Warning] Skipped batch due to generation error:", str(e))
                    batch_outputs.append(["" for _ in range(len(q_batch))])

            for i, q in enumerate(q_batch):
                generations = [extract_answer(batch_outputs[sample_idx][i]) for sample_idx in range(num_samples)]
                generations = [ans for ans in generations if ans]

                if not generations:
                    continue

                final = max(set(generations), key=generations.count)
                gold = gold_batch[i]
                if final == gold:
                    correct += 1
                total += 1
                print('TOTAL = ', total)
                all_generations[q] = generations

            if total % 20 == 0:
                acc = 100 * correct / total
                print(f"[Step {total}] Accuracy: {acc:.2f}%")

    acc = 100 * correct / total
    print(f"\nFinal Accuracy over {total} examples: {acc:.2f}%")

    with open("generations.json", "w") as f:
        json.dump(all_generations, f)
    print("Saved generations to generations.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_eval(
        model_name_or_path=args.model,
        lora_ckpt_path=args.lora_ckpt,
        split=args.split,
        num_samples=args.num_samples,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
