import torch
from ring_attention_pytorch import RingAttention
from torch import Tensor, nn
from zeta.nn import FeedForward, OutputHead


def exists(val):
    return val is not None


class LongGeminiTransformerBlock(nn.Module):
    """
    Gemini15TransformerBlock is a transformer block used in the Gemini15 model.

    Args:
        dim (int): The input dimension of the block.
        depth (int, optional): The depth of the block. Defaults to 32.
        dim_head (int, optional): The dimension of each head in the multi-head attention mechanism. Defaults to 128.
        heads (int, optional): The number of attention heads. Defaults to 24.
        use_abs_pos_emb (bool, optional): Whether to use absolute positional embeddings. Defaults to False.
        attn_flash (bool, optional): Whether to use flash attention. Defaults to True.
        attn_kv_heads (int, optional): The number of heads to use for key-value attention. Defaults to 2.
        qk_norm (bool, optional): Whether to apply layer normalization to query, key, and value. Defaults to True.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feedforward network. Defaults to 4.

    Attributes:
        dim (int): The input dimension of the block.
        depth (int): The depth of the block.
        dim_head (int): The dimension of each head in the multi-head attention mechanism.
        heads (int): The number of attention heads.
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings.
        attn_flash (bool): Whether to use flash attention.
        attn_kv_heads (int): The number of heads to use for key-value attention.
        qk_norm (bool): Whether to apply layer normalization to query, key, and value.
        attn (RingAttention): The attention model for the block.
        norm (nn.LayerNorm): The layer normalization module.
        ffn (FeedForward): The feedforward model for the block.

    """

    def __init__(
        self,
        dim: int,
        depth: int = 32,
        dim_head: int = 128,
        heads: int = 24,
        qk_norm: bool = True,
        ff_mult: int = 4,
        ring_seq_size: int = 512,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.qk_norm = qk_norm
        self.ff_mult = ff_mult
        self.ring_seq_size = ring_seq_size

        # Attention model for the block
        self.attn = RingAttention(
            dim=dim,
            dim_head=dim_head,
            heads=True,
            causal=True,
            auto_shard_seq=True,
            ring_attn=True,
            ring_seq_size=ring_seq_size,
            prenorm=True,
            *args,
            **kwargs,
        )

        # Post Attention layer normalization
        self.norm = nn.LayerNorm(dim)

        # Feedforward model for the block
        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

    def forward(self, x: Tensor, *args, **kwargs):
        """
        Forward pass of the Gemini15TransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x = self.norm(x)

        # Attention
        x = self.attn(x) + x

        # Feedforward
        x = self.ffn(x) + x

        return x


class LongGemini(nn.Module):
    """
    LongGemini model implementation.

    Args:
        dim (int): Dimension of the input.
        depth (int, optional): Depth of the model. Defaults to 32.
        dim_head (int, optional): Dimension of each head. Defaults to 128.
        long_gemini_depth (int, optional): Depth of the LongGemini model. Defaults to 9.
        heads (int, optional): Number of attention heads. Defaults to 24.
        qk_norm (bool, optional): Whether to apply normalization to query and key. Defaults to True.
        ff_mult (int, optional): Multiplier for the feed-forward layer dimension. Defaults to 4.
        ring_seq_size (int, optional): Size of the ring sequence. Defaults to 512.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 32,
        num_tokens: int = 10000,
        seq_len: int = 8192,
        dim_head: int = 128,
        long_gemini_depth: int = 9,
        heads: int = 24,
        qk_norm: bool = True,
        ff_mult: int = 4,
        ring_seq_size: int = 512,
        *args,
        **kwargs,
    ):
        super(LongGemini, self).__init__()
        self.dim = dim
        self.depth = depth
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.dim_head = dim_head
        self.long_gemini_depth = long_gemini_depth
        self.heads = heads
        self.qk_norm = qk_norm
        self.ff_mult = ff_mult
        self.ring_seq_size = ring_seq_size

        self.output_head = OutputHead(
            dim,
            1,
        )

        # Layers for the model
        self.layers = nn.ModuleList(
            [
                LongGeminiTransformerBlock(
                    dim,
                    depth,
                    dim_head,
                    heads,
                    qk_norm,
                    ff_mult,
                    ring_seq_size,
                    *args,
                    **kwargs,
                )
                for _ in range(long_gemini_depth)
            ]
        )

        # Embedding layer for the model
        self.embed = nn.Embedding(num_tokens, dim)

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        text: Tensor,
        *args,
        **kwargs,
    ):
        """
        Forward pass of the LongGemini model.

        Args:
            x (Tensor): Input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: Output tensor.
        """
        # Text embedding
        x = self.embed(text)
        x = self.norm(x)

        # Apply the layers
        for layer in self.layers:
            x = layer(x)
            x = self.norm(x)

        return self.output_head(x)

