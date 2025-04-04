import torch
import torch.nn.functional as F

def quant_matmul(x, weight_int8, scale, zero, bias=None):
    """
    Perform quantized matrix multiplication: (x @ dequantized(weight)) + bias
    Args:
        x: [batch, in_features] (float16/bfloat16)
        weight_int8: [out_features, in_features] (uint8)
        scale: [out_features, in_features] (float32)
        zero: [out_features, in_features] (float32)
        bias: [out_features] (float32 or None)
    Returns:
        output: [batch, out_features]
    """
    x = x.to(torch.float16)
    scale = scale.to(x.device)
    zero = zero.to(x.device)

    # Debugging assertion
    assert weight_int8.shape == scale.shape == zero.shape, (
        f"Shape mismatch: weight_int8={weight_int8.shape}, scale={scale.shape}, zero={zero.shape}"
    )

    # Dequantize
    weight = (weight_int8.to(torch.float16) - zero) * scale
    weight = weight.to(x.device)

    out = F.linear(x, weight, bias)
    return out
