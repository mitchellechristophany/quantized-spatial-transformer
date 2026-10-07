import torch
import torch.nn as nn

class SpatialAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.fc(attn_out)

# Instantiate and test precision casting
model = SpatialAttentionBlock()
dummy_input = torch.randn(10, 32, 128)

# Simulating Mixed-Precision / FP16 Execution
model_fp16 = model.half()
input_fp16 = dummy_input.half()
output = model_fp16(input_fp16)

print("--- Quantized Spatial Transformer ---")
print(f"Input Shape: {input_fp16.shape}")
print(f"Output Shape: {output.shape}")
print(f"Precision State: {output.dtype}")
