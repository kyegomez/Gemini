import torch
from torch import nn
from torch.nn import Module
from zeta.structs import AutoregressiveWrapper

from gemini_torch.transformer import Decoder, Transformer
from einops import rearrange, reduce


def exists(val):
    return val is not None


class Gemini(Module):
    """
    Gemini model class.


    Args:
    - num_tokens: Number of tokens in the vocabulary
    - max_seq_len: Maximum sequence length
    - dim: Dimension of the model
    - depth: Depth of the model
    - dim_head: Dimension of the model head
    - heads: Number of heads
    - use_abs_pos_emb: Whether to use absolute position embedding
    - alibi_pos_bias: Alibi position bias
    - alibi_num_heads: Number of alibi heads
    - rotary_xpos: Rotary position
    - attn_flash: Attention flash
    - deepnorm: Deep normalization
    - shift_tokens: Number of tokens to shift
    - attn_one_kv_head: Attention one key/value head
    - qk_norm: Query-key normalization
    - attn_qk_norm: Attention query-key normalization
    - attn_qk_norm_dim_scale: Attention query-key normalization dimension scale
    - embedding_provider: Embedding provider module

    """

    def __init__(
        self,
        num_tokens=50432,
        max_seq_len=32052,
        dim=2560,
        depth=32,
        dim_head=128,
        heads=24,
        use_abs_pos_emb=False,
        attn_flash=True,
        attn_kv_heads=2,
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True,
        patches: int = 16,
        patch_size: int = 16,
        img_channels: int = 3,
        audio_seq_len: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.use_abs_pos_emb = use_abs_pos_emb
        self.attn_flash = attn_flash
        self.attn_kv_heads = attn_kv_heads
        self.qk_norm = qk_norm
        self.attn_qk_norm = attn_qk_norm
        self.attn_qk_norm_dim_scale = attn_qk_norm_dim_scale
        self.patches = patches
        self.patch_size = patch_size
        self.img_channels = img_channels
        self.audio_seq_len = audio_seq_len

        # Transformer model for the model
        self.gemini = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                attn_flash=attn_flash,
                attn_kv_heads=attn_kv_heads,
                qk_norm=qk_norm,
                attn_qk_norm=attn_qk_norm,
                attn_qk_norm_dim_scale=attn_qk_norm_dim_scale,
                cross_attend=True,
                *args,
                **kwargs,
            ),
        )

        # Autoregressive wrapper for the model
        self.decoder = AutoregressiveWrapper(self.gemini)

    def forward(
        self,
        text: torch.Tensor = None,
        img: torch.Tensor = None,
        audio: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """
        Forward pass of the model.

        Args:
        - text: Text tensor
        - img: Image tensor

        Returns:
        - torch.Tensor: The output of the model

        Text input shape: [batch, seq_len, dim]
        img input shape: [batch, channels, height, width]
        audio input shape: [batch, audio_seq_len]

        Output shape: [batch, seq_len, dim]


        """
        print(f"Text: {text.shape} and text dtype: {text.dtype}")

        # Audio dimensions
        img_b, img_c, img_h, img_w = img.shape

        img_to_text = reduce(img, "b c h w -> b c (h w)", "mean")
        img_proj = nn.Linear(img_h * img_w, self.dim)
        img = img_proj(img_to_text)
        # Reshape to apply the linear on the last dimension c to make it compatible
        img = rearrange(img, "b c d -> b d c")
        two_proj = nn.Linear(img_c, self.max_seq_len)
        img = two_proj(img)
        img = rearrange(img, "b d c -> b c d")

        
        
        ########## Audio ##########
        # Audio transformations to add a 3rd dimension
        audio_3d = rearrange(audio, "b l -> b l 1")
        print(f"Audio 3d: {audio_3d.shape}")

        # Audio dimensions
        audio_b, audio_seq_len, audio_dim = audio_3d.shape

        # Audio proj last dimension
        audio_proj = nn.Linear(audio_dim, self.dim)
        audio = audio_proj(audio_3d)
        print(f"Audio proj shape: {audio.shape}")

        # Audio reshape seqlen
        audio = rearrange(audio, "b l d -> b d l")
        audio_proj2 = nn.Linear(audio_seq_len, self.max_seq_len)
        audio = audio_proj2(audio)
        audio = rearrange(audio, "b d l -> b l d")
        print(f"Audio final shape: {audio.shape}")

        # Fuse layers
        fused = torch.cat((img, audio), dim=1)

        # audio
        # print(img_to_text.shape)
        # fused = torch.concat((img, audio))
        return self.decoder(text, context=fused, *args, **kwargs)
