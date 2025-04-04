# Efficient Reasoning LLM: Full Pipeline

This project implements a complete and modular pipeline to fine-tune, quantize, and deploy LLMs for math and commonsense reasoning.

## Features
- Manual LoRA injection and training
- GPTQ-style post-training quantization (groupwise, Hessian-aware)
- Chain-of-Thought (CoT) generation + reranking
- Verifier-based output scoring
- GSM8K + CommonsenseQA task support

## File Structure
```
.
├── lora.py                  # Custom LoRA injection
├── finetune.py              # LoRA adapter training
├── dataset_loader.py        # GSM8K CoT formatting
├── inject_lora.py           # Merge LoRA weights
├── gptq_quantizer.py        # GPTQ quantization
├── evaluate_qa.py           # Answer accuracy metrics
├── inference_rerank.py      # CoT sampling + reranking
├── verifier_model.py        # Small BERT-based verifier
├── commonsenseqa_loader.py  # Multi-choice QA formatting
├── quant_matmul.py          # Simulated int4 × fp16 kernel
├── gsm8k_bootstrap.py       # Generate few-shot CoT examples
├── run_pipeline.py          # End-to-end CLI pipeline
├── demo_notebook.ipynb      # Interactive notebook demo
```

## Quick Start
### 1. Train LoRA adapters:
```bash
python finetune.py
```

### 2. Merge and quantize:
```bash
python inject_lora.py
python run_pipeline.py --eval_mode rerank
```

### 3. Bootstrap data:
```bash
python gsm8k_bootstrap.py
```

## Model Requirements
- LLaMA 2 or Mistral 7B (available via Hugging Face)
- GPU with ≥24GB for training, ≥12GB for quantized inference

## Tasks Supported
- GSM8K (math)
- CommonsenseQA (multi-choice)

## License
MIT. Created by Rishikesh Ksheersagar.

---

> This repo is designed for researchers and engineers who want full control over fine-tuning, quantization, and reasoning workflows without black-box dependencies.

