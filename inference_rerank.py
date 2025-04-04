import torch
from transformers import AutoTokenizer
from dataset_loader import GSM8KCoTDataset
from evaluate_qa import extract_final_answer, compute_math_accuracy
import torch.nn.functional as F
from torch.utils.data import DataLoader


def generate_candidates(model, tokenizer, inputs, k=4, max_new_tokens=128):
    input_ids = inputs["input_ids"].unsqueeze(0).to(model.device)
    attention_mask = inputs["attention_mask"].unsqueeze(0).to(model.device)

    generations = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=k
    )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generations]


def simple_verifier_score(answer_text):
    score = 0.0
    if "answer is" in answer_text.lower():
        score += 0.5
    if extract_final_answer(answer_text):
        score += 0.5
    return score


def rerank_candidates(candidates):
    return sorted(candidates, key=lambda x: simple_verifier_score(x), reverse=True)[0]


def evaluate_reranked(model, tokenizer, dataset, k=4):
    model.eval()
    preds, targets = [], []

    for item in dataset:
        candidates = generate_candidates(model, tokenizer, item, k=k)
        best = rerank_candidates(candidates)

        pred_ans = extract_final_answer(best)
        true_ans = extract_final_answer(tokenizer.decode(item["labels"], skip_special_tokens=True))

        if pred_ans and true_ans:
            preds.append(pred_ans)
            targets.append(true_ans)

    acc = compute_math_accuracy(preds, targets)
    print(f"[Rerank] Accuracy with k={k}: {acc * 100:.2f}%")
    return acc
