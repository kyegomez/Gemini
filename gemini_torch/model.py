import torch
from torch.nn import Module
from zeta.structs import AutoregressiveWrapper

from gemini_torch.transformer import Decoder, Transformer
from gemini_torch.utils import ImgToTransformer, AudioToLangEmbedding


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
        alibi_pos_bias=True,
        alibi_num_heads=12,
        rotary_xpos=True,
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
        **kwargs
    ):
        super().__init__()

        try:
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
                    alibi_pos_bias=alibi_pos_bias,
                    alibi_num_heads=alibi_num_heads,
                    rotary_xpos=rotary_xpos,
                    attn_flash=attn_flash,
                    attn_kv_heads=attn_kv_heads,
                    qk_norm=qk_norm,
                    attn_qk_norm=attn_qk_norm,
                    attn_qk_norm_dim_scale=attn_qk_norm_dim_scale,
                    *args,
                    **kwargs
                ),
            )

            # Autoregressive wrapper for the model
            self.decoder = AutoregressiveWrapper(self.gemini)

            # Takes in imgs -> patches them -> transforms them to the same dimension as the model
            self.img_to_transformer = ImgToTransformer(
                patches=patches,
                patch_size=patch_size,
                transformer_dim=dim,
                img_channels=img_channels,
                seq_len=num_tokens,
                reduced_dim=dim,
                *args,
                **kwargs
            )

            # Takes in audio -> transforms it to the same dimension as the model
            self.audio_to_lang_embedding = AudioToLangEmbedding(
                audio_seq_len=audio_seq_len, seqlen=num_tokens, dim=dim, *args, **kwargs
            )

        except Exception as e:
            print("Failed to initialize gemini: ", e)
            raise

    def forward(
        self,
        text: torch.Tensor,
        img: torch.Tensor = None,
        audio: torch.Tensor = None,
        *args,
        **kwargs
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
        try:
            # If image is provided, concat it with the text
            if exists(img):
                img = self.img_to_transformer(img)
                x = torch.concat((text, img), dim=1)
                model_input = self.decoder.forward(x)[0]

            # Else, just use the text
            elif exists(img) and exists(audio):
                # Concat the audio and image and text embeddings all at once, audio is [batch, audio_seq_len]
                x = self.audio_to_lang_embedding(audio)
                x = self.img_to_transformer(img)
                x = torch.concat((text, img, audio), dim=1)
                model_input = self.decoder.forward(x)[0]

            else:
                model_input = self.decoder.forward(text)[0]

            return self.decoder(model_input, padded_x=model_input[0])
        except Exception as e:
            print("Failed in forward method: ", e)
            raise
