from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import os


def build_fewshot_prompt(example, fewshots):
    prompt = "You are a math expert who explains solutions step by step.\n"
    for fs in fewshots:
        prompt += f"\n### Question:\n{fs['question']}\n\n### Answer:\n{fs['answer']}\n"
    prompt += f"\n### Question:\n{example['question']}\n\n### Answer:\n"
    return prompt


def generate_cot_examples(model, tokenizer, split="train", num_samples=500, fewshot_k=3):
    ds = load_dataset("gsm8k", "main", split=split)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    examples = []

    fewshots = ds.select(range(fewshot_k))  # first k examples as few-shot context

    for i in range(num_samples):
        ex = ds[i + fewshot_k]  # avoid overlap
        fs_data = [
            {"question": fewshots[j]["question"], "answer": fewshots[j]["answer"]}
            for j in range(fewshot_k)
        ]
        prompt = build_fewshot_prompt(ex, fs_data)

        gen = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"]
        gen_answer = gen.split("### Answer:")[-1].strip()

        examples.append({
            "question": ex["question"],
            "generated_answer": gen_answer,
            "original_answer": ex["answer"]
        })

        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{num_samples} examples")

    os.makedirs("bootstrap_data", exist_ok=True)
    with open("bootstrap_data/gsm8k_fewshot_cot.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved bootstrapped CoT examples to bootstrap_data/")


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    generate_cot_examples(model, tokenizer, num_samples=100, fewshot_k=3)
