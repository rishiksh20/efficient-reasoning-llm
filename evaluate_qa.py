import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset_loader import GSM8KCoTDataset
import re


def extract_final_answer(text):
    match = re.search(r"(?:answer is|Answer:|Final Answer:|####)\s*(-?[0-9\.]+)", text)
    return match.group(1).strip() if match else None


def compute_math_accuracy(preds, targets):
    correct = 0
    total = len(preds)
    for p, t in zip(preds, targets):
        try:
            if float(p) == float(t):
                correct += 1
        except:
            continue
    return correct / total if total > 0 else 0.0


def evaluate_model(model, tokenizer, dataset, max_new_tokens=128, batch_size=4):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1
            )

            for gen, label in zip(outputs, batch["labels"]):
                pred_text = tokenizer.decode(gen, skip_special_tokens=True)
                label_text = tokenizer.decode(label, skip_special_tokens=True)

                pred_ans = extract_final_answer(pred_text)
                true_ans = extract_final_answer(label_text)

                if pred_ans and true_ans:
                    preds.append(pred_ans)
                    targets.append(true_ans)

    acc = compute_math_accuracy(preds, targets)
    print(f"[Eval] Math Accuracy: {acc * 100:.2f}%")
    return acc
