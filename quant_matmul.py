import torch
import torch.nn as nn

class QLinear(nn.Module):
    """
    Simulated quantized linear layer: W_int4 * scale
    """
    def __init__(self, weight_int4, scale):
        super().__init__()
        self.register_buffer("W_q", weight_int4)  # shape [out, in] int8 simulating int4
        self.register_buffer("scale", scale)      # shape [out]

    def forward(self, x):
        W_fp = self.W_q.float() * self.scale.view(-1, 1)  # dequantize row-wise
        return torch.matmul(x, W_fp.T)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, in_features, out_features = 2, 16, 4
    X = torch.randn(batch, in_features)
    W_real = torch.randn(out_features, in_features)
    scale = (2 ** 4 - 1) / W_real.abs().max(dim=1).values
    W_q = (W_real * scale.view(-1, 1)).round().clamp(-8, 7).to(torch.int8)

    qlinear = QLinear(W_q, scale)
    Y = qlinear(X)

    print("[Quant] Simulated output:", Y)
