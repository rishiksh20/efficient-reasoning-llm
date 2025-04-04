import torch
import torch.nn as nn
from collections import defaultdict
from quant_matmul import QLinear

class GPTQQuantizer:
    def __init__(self, model, layer_names, bits=4, group_size=64):
        self.model = model.eval()
        self.layer_names = layer_names
        self.bits = bits
        self.group_size = group_size
        self.activation_cache = defaultdict(list)
        self.quantized_weights = {}
        self.scales = {}

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                module.register_forward_hook(self._capture_hook(name))

    def _capture_hook(self, name):
        def hook(module, input, output):
            self.activation_cache[name].append(input[0].detach().cpu())
        return hook

    def collect_activations(self, dataloader, device="cuda", max_batches=50):
        self._register_hooks()
        self.model.to(device)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _approximate_hessian(self, A):
        # A is (N x d), we return (d x d)
        return A.T @ A

    def quantize_layer(self, layer_name):
        layer = dict(self.model.named_modules())[layer_name]
        W = layer.weight.data.clone().cpu()  # (out x in)
        A = torch.cat(self.activation_cache[layer_name], dim=0).cpu()  # (N x in)

        H = self._approximate_hessian(A)  # (in x in)
        W_q = torch.zeros_like(W)
        scales = torch.zeros(W.size(0))

        for i in range(0, W.size(0), self.group_size):
            block_rows = W[i:i+self.group_size]
            block_H = H + 1e-4 * torch.eye(H.size(0))  # regularization

            try:
                L = torch.linalg.cholesky(block_H)
            except:
                L = torch.eye(H.size(0))

            for j in range(block_rows.size(0)):
                idx = i + j
                w_i = block_rows[j].clone()
                max_val = w_i.abs().max()

                if max_val < 1e-5:
                    scales[idx] = 1.0
                    W_q[idx] = 0.0
                    continue

                s_i = (2 ** self.bits - 1) / max_val
                q_i = torch.clamp((w_i * s_i).round(), -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1)
                W_q[idx] = q_i / s_i
                scales[idx] = s_i

                # Error feedback
                error = w_i - W_q[idx]
                correction = torch.cholesky_solve(error.unsqueeze(1), L).squeeze(1)
                if j + 1 < block_rows.size(0):
                    block_rows[j + 1] += correction * 0.1

        self.quantized_weights[layer_name] = W_q
        self.scales[layer_name] = scales

    def apply_quantization(self):
        for name in self.layer_names:
            self.quantize_layer(name)
            layer = dict(self.model.named_modules())[name]

            qlinear = QLinear(
                self.quantized_weights[name].to(layer.weight.device).to(torch.int8),
                self.scales[name].to(layer.weight.device)
            )
            parent = self.model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], qlinear)

    def save(self, path="quantized_model.pt"):
        torch.save({
            "quantized_weights": self.quantized_weights,
            "scales": self.scales
        }, path)
        print(f"[GPTQ] Quantized weights saved to {path}")
