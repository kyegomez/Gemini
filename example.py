import torch
from gemini_torch.model import Gemini

# Initialize model with smaller dimensions
model = Gemini(
    num_tokens=10000,  # Reduced from 50432
    max_seq_len=1024,  # Reduced from 4096
    dim=320,  # Reduced from 1280
    depth=8,  # Reduced from 16
    dim_head=32,  # Reduced from 64
    heads=6,  # Reduced from 12
    use_abs_pos_emb=False,
    attn_flash=True,
    attn_kv_heads=2,
    qk_norm=True,
    attn_qk_norm=True,
    attn_qk_norm_dim_scale=True,
    post_fusion_norm=True,
    post_modal_transform_norm=True,
)

# Text shape: [batch, seq_len, dim]
text = torch.randint(0, 10000, (1, 1024))  # Reduced seq_len from 4096

# Img shape: [batch, channels, height, width]
img = torch.randn(1, 3, 64, 64)  # Reduced height and width from 128

# Audio shape: [batch, audio_seq_len, dim]
audio = torch.randn(1, 32)  # Reduced audio_seq_len from 64

# Apply model to text and img
y, _ = model(text=text, img=img, audio=audio)

# Output shape: [batch, seq_len, dim]
print(y)
print(y.shape)
