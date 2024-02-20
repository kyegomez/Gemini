import torch
from gemini_torch.model import LongGemini

# Text tokens
x = torch.randint(0, 10000, (1, 1024))

model = LongGemini(
    dim=512,  # Dimension of the input tensor
    depth=32,  # Number of transformer blocks
    dim_head=128,  # Dimension of the query, key, and value vectors
    long_gemini_depth=9,  # Number of long gemini transformer blocks
    heads=24,  # Number of attention heads
    qk_norm=True,  # Whether to apply layer normalization to query and key vectors
    ring_seq_size=512,  # The size of the ring sequence
)

out = model(x)  # Apply the model to the input tensor

print(out)  # Print the output tensor
