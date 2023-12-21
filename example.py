import torch
from gemini_torch import Gemini

# Initialize the model
model = Gemini(
    num_tokens=12608,
    max_seq_len=2048,
    dim=640,
    depth=8,
    dim_head=32,
    heads=6,
    use_abs_pos_emb=False,
    alibi_pos_bias=True,
    alibi_num_heads=3,
    rotary_xpos=True,
    attn_flash=True,
    attn_kv_heads=1,
    qk_norm=True,
    attn_qk_norm=True,
    attn_qk_norm_dim_scale=True,
)

# Initialize the text random tokens
x = torch.randint(0, 12608, (1, 2048))

# Apply model to x
y = model(x)

# Print logits
print(y)
