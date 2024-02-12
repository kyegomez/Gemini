import torch
from torch import nn
from torch.nn import Module
from zeta.structs import AutoregressiveWrapper

from gemini_torch.transformer import Decoder, Transformer
from zeta.nn import (
    audio_to_text,
    video_to_text,
    img_to_text,
)


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
        post_fusion_norm: bool = True,
        post_modal_transform_norm: bool = False,
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
        self.post_fusion_norm = post_fusion_norm
        self.post_modal_transform_norm = post_modal_transform_norm

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

        # Post fusion norm
        if self.post_fusion_norm:
            self.psf_norm = nn.LayerNorm(dim)

        if self.post_modal_transform_norm:
            self.pmt_norm = nn.LayerNorm(dim)

    def forward(
        self,
        text: torch.Tensor = None,
        img: torch.Tensor = None,
        audio: torch.Tensor = None,
        video: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """
        Forward pass of the model.

        Args:
        - text: Text tensor
        - img: Image tensor
        - audio: Audio tensor
        - video: Video tensor

        Returns:
        - torch.Tensor: The output of the model

        Text input shape: [batch, seq_len, dim]
        img input shape: [batch, channels, height, width]
        audio input shape: [batch, audio_seq_len]
        video input shape: [batch, channels, frames, height, width]

        Output shape: [batch, seq_len, dim]
        """
        assert (
            (img is not None and audio is not None)
            or (img is not None and video is not None)
            or (audio is not None and video is not None)
        ), "At least two of the inputs (img, audio, video) must be provided."

        if img is not None:
            # Image dimensions
            img_b, img_c, img_h, img_w = img.shape

            # img = img_to_text(img, self.patches, self.patch_size, self.dim, True)
            img = img_to_text(img, self.max_seq_len, self.dim, True)

            if self.post_modal_transform_norm:
                img = self.pmt_norm(img)

        if audio is not None:
            # Audio dimensions
            audio_b, audio_seq_len = audio.shape

            audio = audio_to_text(audio, self.max_seq_len, self.dim, True)

            if self.post_modal_transform_norm:
                audio = self.pmt_norm(audio)

        if video is not None:
            # Video dimensions
            video_b, video_c, video_f, video_h, video_w = video.shape

            video = video_to_text(video, self.max_seq_len, self.dim, True)

        # Fuse layers
        if img is not None and audio is not None:
            fused = torch.cat((img, audio), dim=1)
        elif img is not None and video is not None:
            fused = torch.cat((img, video), dim=1)
        elif audio is not None and video is not None:
            fused = torch.cat((audio, video), dim=1)

        # Post fusion layernorm for stability.
        if self.post_fusion_norm:
            fused = self.psf_norm(fused)

        return self.decoder(text, context=fused, *args, **kwargs)
