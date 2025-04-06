import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_matmul import quant_matmul

class GPTQQuantizer:
    def __init__(self, layer, block_size=64, bits=8):
        self.layer = layer
        self.block_size = block_size
        self.bits = bits
        self.max_int = 2 ** bits - 1
        self.device = layer.weight.device

        self.W = layer.weight.data.clone().to(torch.float32)
        self.in_features = self.W.shape[1]
        self.out_features = self.W.shape[0]

        self.H = torch.zeros((self.in_features, self.in_features), dtype=torch.float32, device=self.device)
        self.A = []

    def add_batch(self, inputs):
        inputs = inputs.view(-1, inputs.shape[-1])
        self.A.append(inputs.detach())

    def compute_hessian(self):
        A = torch.cat(self.A, dim=0)
        self.H = (A.T @ A).float()
        self.H += torch.eye(self.H.shape[0], device=self.device, dtype=torch.float32) * 1e-5  # Regularization

    def quantize(self):
        W = self.W
        H = self.H
        block_size = self.block_size

        W_quant = torch.zeros_like(W)
        scale_list = []
        zero_list = []

        for i in range(0, W.shape[1], block_size):
            end = min(i + block_size, W.shape[1])
            W_block = W[:, i:end]  # (out, block)

            # H_block = H[i:end, i:end]
            H_block = H[i:end, i:end].float()
            try:
                H_inv = torch.linalg.inv(H_block)
            except:
                H_inv = torch.pinverse(H_block)

            proj = W_block @ H_inv @ H_block  # projection

            W_min = proj.min(dim=1, keepdim=True)[0]
            W_max = proj.max(dim=1, keepdim=True)[0]
            scale = (W_max - W_min) / self.max_int
            scale[scale == 0] = 1e-6
            zero = torch.round(-W_min / scale)
            W_q = torch.round(proj / scale + zero).clamp(0, self.max_int)

            dequant = (W_q - zero) * scale
            residual = W_block - dequant

            # Backsolve error propagation
            if end < W.shape[1]:
                W[:, end:] -= residual @ H[i:end, end:]

            W_quant[:, i:end] = W_q

            scale = scale.expand(-1, W_block.shape[1])
            zero = zero.expand(-1, W_block.shape[1])

            scale_list.append(scale)
            zero_list.append(zero)

        self.W_int8 = W_quant.to(torch.uint8)
        self.scales = torch.cat(scale_list, dim=1)
        self.zeros = torch.cat(zero_list, dim=1)

        return self.W_int8, self.scales, self.zeros


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def load_quantized(self, W_int8, scales, zeros):
        self.register_buffer("weight_quant", W_int8)
        self.register_buffer("scale", scales)
        self.register_buffer("zero", zeros)

    def forward(self, x):
        return quant_matmul(x, self.weight_quant, self.scale, self.zero, bias=self.bias)


@torch.no_grad()
def quantize_model_weights(model, dataloader, num_batches=10):
    print("[GPTQ] Starting full quantization...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear): #and any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            print(f"Quantizing {name}...")
            quantizer = GPTQQuantizer(module)

            count = 0
            for batch in dataloader:
                if count >= num_batches:
                    break
                input_ids = batch["input_ids"].to(module.weight.device)


                # with torch.no_grad():
                hidden = model.model.embed_tokens(input_ids)


                quantizer.add_batch(hidden)
                count += 1

            quantizer.compute_hessian()
            W_q, scales, zeros = quantizer.quantize()

            qlinear = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
            qlinear.load_quantized(W_q, scales, zeros)
            if module.bias is not None:
                qlinear.bias.data.copy_(module.bias.data)

            # Save original float weight for LoRA injection
            # qlinear.fp_weight = quantizer.W.clone()  # Save original W

            # Replace original layer
            parent = dict(model.named_modules())[name.rsplit(".", 1)[0]]
            setattr(parent, name.rsplit(".", 1)[-1], qlinear)

    return model
