import torch
import torch.nn.functional as F

def quant_matmul(x, weight_int, scale, zero, bias=None):
    """
    Perform quantized matrix multiplication: (x @ dequantized(weight)) + bias

    Args:
        x: [batch, in_features] (float16/bfloat16)
        weight_int: [out_features, in_features] (uint8) OR packed 4-bit weights
        scale: [out_features] (float32/float16)
        zero:  [out_features] (float32/float16)
        bias:  [out_features] or None

    Returns:
        output: [batch, out_features]
    """
    x = x.to(torch.float16)
    device = x.device
    scale = scale.to(device).unsqueeze(1)
    zero = zero.to(device).unsqueeze(1)

    out_features, compressed_dim = weight_int.shape
    in_features = x.shape[-1]

    if compressed_dim * 2 == in_features:
        # We are dealing with 4-bit packed weights, unpack first
        unpacked = torch.empty((out_features, in_features), dtype=torch.uint8, device=device)
        unpacked[:, 0::2] = weight_int >> 4
        unpacked[:, 1::2] = weight_int & 0x0F
        weight = (unpacked.to(torch.float16) - zero) * scale
    else:
        # Normal 8-bit quantization
        weight = (weight_int.to(torch.float16) - zero) * scale

    weight = weight.to(device)

    return F.linear(x, weight, bias)