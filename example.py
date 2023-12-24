import torch
from gemini_torch.model import Gemini

# Initialize model with smaller dimensions
model = Gemini(
    num_tokens=50432,
    max_seq_len=4096,  # Reduced from 8192
    dim=1280,  # Reduced from 2560
    depth=16,  # Reduced from 32
    dim_head=64,  # Reduced from 128
    heads=12,  # Reduced from 24
    use_abs_pos_emb=False,
    attn_flash=True,
    attn_kv_heads=2,
    qk_norm=True,
    attn_qk_norm=True,
    attn_qk_norm_dim_scale=True,
)

# Text shape: [batch, seq_len, dim]
text = torch.randint(0, 50432, (1, 4096))  # Reduced seq_len from 8192

# Img shape: [batch, channels, height, width]
img = torch.randn(1, 3, 128, 128)  # Reduced height and width from 256

# Audio shape: [batch, audio_seq_len, dim]
audio = torch.randn(1, 64)  # Reduced audio_seq_len from 128

# Apply model to text and img
y = model(text, img, audio)

# Output shape: [batch, seq_len, dim]
print(y)
