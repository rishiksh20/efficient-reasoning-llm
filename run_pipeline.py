import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_loader import GSM8KCoTDataset
from commonsenseqa_loader import CommonsenseQADataset
from finetune import train_lora_model
from inject_lora import merge_lora_weights
from gptq_quantizer import GPTQQuantizer
from evaluate_qa import evaluate_model
from inference_rerank import evaluate_reranked
from inference_consistency import generate_consistent_answers, VerifierWrapper
from verifier_model import VerifierModel, train_verifier
from lora import inject_lora_layers


def run_pipeline(model_name, eval_mode="simple", task="gsm8k", k=4, verifier=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    if task == "gsm8k":
        dataset = GSM8KCoTDataset(split="test", tokenizer=tokenizer)
    elif task == "commonsenseqa":
        dataset = CommonsenseQADataset(split="validation", tokenizer=tokenizer)
    else:
        raise ValueError("Unsupported task")

    print("[1] Loading and injecting LoRA...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = inject_lora_layers(model)
    model.load_state_dict(torch.load("checkpoints/lora_adapter.pt"), strict=False)

    print("[2] Merging LoRA...")
    model = merge_lora_weights(model)
    torch.save(model.state_dict(), "checkpoints/merged_lora_model.pt")

    print("[3] Quantizing with GPTQ...")
    layer_names = [n for n, _ in model.named_modules() if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj"])]
    quantizer = GPTQQuantizer(model, layer_names=layer_names)
    quantizer.collect_activations(torch.utils.data.DataLoader(dataset, batch_size=2))
    quantizer.apply_quantization()
    quantizer.save("checkpoints/quantized_model.pt")

    print("[4] Evaluating...")
    model.to("cuda")
    if eval_mode == "simple":
        evaluate_model(model, tokenizer, dataset)
    elif eval_mode == "rerank":
        evaluate_reranked(model, tokenizer, dataset, k=k)
    elif eval_mode == "consistency":
        verifier_model = VerifierWrapper("checkpoints/verifier_model", tokenizer) if verifier else None
        generate_consistent_answers(model, tokenizer, dataset, k=k, verifier=verifier_model)

    if verifier and eval_mode != "consistency":
        print("[5] Training verifier...")
        candidates = ["The answer is 72.", "The answer is 0."] * 20
        labels = [1, 0] * 20
        verifier_model = VerifierModel()
        train_verifier(verifier_model, tokenizer, candidates, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--eval_mode", choices=["simple", "rerank", "consistency"], default="simple")
    parser.add_argument("--task", choices=["gsm8k", "commonsenseqa"], default="gsm8k")
    parser.add_argument("--k", type=int, default=4, help="number of generations or candidates")
    parser.add_argument("--verifier", action="store_true", help="enable verifier reranking")
    args = parser.parse_args()

    run_pipeline(args.model, args.eval_mode, args.task, args.k, verifier=args.verifier)
