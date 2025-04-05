# full_eval.py
import torch
import json
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from lora import inject_lora_layers
from merge_lora import merge_lora_weights_into_base

SYSTEM_PROMPT = "You are a math expert who explains solutions step by step."
QUESTION_TEMPLATE = "\n### Question:\n{question}\n\n### Answer:\n"

def format_prompt(q):
    return SYSTEM_PROMPT + QUESTION_TEMPLATE.format(question=q.strip())

def extract_answer(text):
    match = re.search(r"The answer is (.*?)(\.|$)", text)
    return match.group(1).strip() if match else None

def score_with_verifier(verifier, verifier_tokenizer, prompt_outputs, device):
    scores = []
    for text in prompt_outputs:
        inputs = verifier_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = verifier(**inputs).logits
            score = torch.softmax(logits, dim=-1)[0][1].item()  # probability of "correct"
            scores.append(score)
    return scores

def run_full_eval(base_model_path, lora_ckpt_path, verifier_path, split="test", num_samples=5, top_p=0.95, temperature=0.7, max_new_tokens=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    model = inject_lora_layers(model)
    model.load_state_dict(torch.load(lora_ckpt_path), strict=False)
    model = merge_lora_weights_into_base(model)  # merges adapters into base weights
    model.to(device).eval()

    verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_path)
    verifier = AutoModelForSequenceClassification.from_pretrained(verifier_path).to(device).eval()

    dataset = load_dataset("gsm8k", "main", split=split)

    correct = 0
    total = 0

    for ex in tqdm(dataset):
        q = ex["question"]
        prompt = format_prompt(q)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)

        generations, full_texts = [], []
        for _ in range(num_samples):
            output = model.generate(
                input_ids,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            cot = decoded[len(prompt):].strip()
            generations.append(cot)
            full_texts.append(prompt + cot)

        scores = score_with_verifier(verifier, verifier_tokenizer, full_texts, device)
        best_idx = scores.index(max(scores))
        final_answer = extract_answer(generations[best_idx])
        gold_answer = ex["answer"].split("####")[-1].strip()

        if final_answer == gold_answer:
            correct += 1
        total += 1

        if total % 20 == 0:
            print(f"[Step {total}] Accuracy: {100 * correct / total:.2f}%")

    acc = 100 * correct / total
    print(f"\nFinal Accuracy over {total} examples: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="LoRA checkpoint")
    parser.add_argument("--verifier", type=str, required=True, help="Verifier model path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    run_full_eval(
        base_model_path=args.model,
        lora_ckpt_path=args.lora_ckpt,
        verifier_path=args.verifier,
        split=args.split,
        num_samples=args.num_samples,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
