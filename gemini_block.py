import torch
from gemini_torch.model import LongGeminiTransformerBlock

# Define input tensor
x = torch.randn(1, 1024, 512)

# Create an instance of Gemini15TransformerBlock
model = LongGeminiTransformerBlock(
    dim=512,  # Dimension of the input tensor
    depth=32,  # Number of transformer blocks
    dim_head=128,  # Dimension of the query, key, and value vectors
    heads=24,  # Number of attention heads
    qk_norm=True,  # Whether to apply layer normalization to query and key vectors
    # ring_seq_size=512,  # The size of the ring sequence
)

# Forward pass
out = model(x)  # Apply the model to the input tensor
print(out)  # Print the output tensor
