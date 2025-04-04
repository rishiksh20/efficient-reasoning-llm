import torch
from collections import Counter
from transformers import AutoModelForSequenceClassification
from evaluate_qa import extract_answer


def generate_consistent_answers(model, tokenizer, dataset, k=5, verifier=None):
    model.eval()
    model.to("cuda")

    correct = 0
    for i, item in enumerate(dataset):
        input_ids = item["input_ids"].unsqueeze(0).to("cuda")
        attention_mask = item["attention_mask"].unsqueeze(0).to("cuda")

        generations = tokenizer.batch_decode(
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                num_return_sequences=k,
                temperature=0.7
            ),
            skip_special_tokens=True
        )

        answers = [extract_answer(gen) for gen in generations]
        counts = Counter(answers)

        if verifier:
            scores = verifier.score(generations)
            top_idx = torch.tensor(scores).argmax().item()
            final_answer = extract_answer(generations[top_idx])
        else:
            final_answer = counts.most_common(1)[0][0]

        gold_answer = extract_answer(tokenizer.decode(item["labels"], skip_special_tokens=True))

        if final_answer == gold_answer:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"[{i+1}] Acc so far: {correct}/{i+1} = {correct / (i+1) * 100:.2f}%")

    print(f"[Self-Consistency] Final Accuracy: {correct / len(dataset) * 100:.2f}%")


class VerifierWrapper:
    def __init__(self, model_path, tokenizer):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
        self.tokenizer = tokenizer
        self.model.eval()

    def score(self, texts):
        with torch.no_grad():
            batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
            scores = self.model(**batch).logits.squeeze(-1).sigmoid().tolist()
            return scores