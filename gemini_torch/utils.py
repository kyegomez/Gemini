import torch
from torch import nn
from einops import rearrange


class ImgToTransformer(nn.Module):
    """ImgToTransformer

    Args:
        patches (int): Number of patches to divide the image into
        patch_size (int): Size of the patches
        transformer_dim (int): Dimension of the transformer
        img_channels (int): Number of channels in the image
        seq_len (int): Length of the sequence
        reduced_dim (int): Dimension of the reduced embedding

    Returns:
        torch.Tensor: The output of the model

    Input shape:
        (batch, channels, height, width)

    Output shape:
        (batch, seq_len, reduced_dim)

    Example:
        >>> import torch
        >>> from geminix import ImgToTransformer
        >>> model = ImgToTransformer(
        ...     patches=16,
        ...     patch_size=16,
        ...     transformer_dim=512,
        ...     img_channels=3,
        ...     seq_len=128,
        ...     reduced_dim=128
        ... )
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)
        >>> y.shape
        torch.Size([1, 128, 128])
    """

    def __init__(
        self,
        patches: int,
        patch_size: int,
        transformer_dim: int,
        img_channels: int,
        seq_len: int,
        reduced_dim: int,
        *args,
        **kwargs,
    ):
        super(ImgToTransformer, self).__init__()
        self.patches = patches
        self.patch_size = patch_size
        self.transformer_dim = transformer_dim
        self.img_channels = img_channels
        self.seq_len = seq_len
        self.reduced_dim = reduced_dim

        # Img is a square, cal number of apthces
        self.num_patches_side = int(patches**0.5)

        # Patch embedding layer
        self.patch_embedding = nn.Linear(
            patch_size * patch_size * img_channels, transformer_dim
        )

        # Dim reduction
        self.dim_reduction = nn.Linear(transformer_dim, reduced_dim)

        # Batch Norm and relu
        self.norm = nn.BatchNorm1d(patches)
        self.activate = nn.ReLU()

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, patches, reduced_dim))

        # Token mixing
        self.token_mixer = nn.Linear(patches * reduced_dim, patches * reduced_dim)

        # Linear layer to expand the seq to vocab
        self.seq_expansion = nn.Linear(patches * reduced_dim, seq_len * reduced_dim)

    def forward(self, x: torch.Tensor):
        """Forward pass

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        batch, channels, height, width, height = x.shape

        # Check if img can be evenly divided into patches
        assert (
            height % self.num_patches_side == 0 and width % self.num_patches_side == 0
        ), f"Image dimensions must be divisivle by the square root of patches"

        # Reshpe the img to patches
        x = x.unfold(
            2,
            self.patch_size,
        ).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch, channels, self.num_patches, -1)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, self.num_patches, -1)

        # Apply patch embedding
        x = self.patch_embedding(x)

        # Dim reduction
        x = self.dim_reduction(x)

        # Batch norm
        x = self.norm(x)
        x = self.activate(x)

        # Add positional encoding
        x = x.view(batch, -1)
        x = self.token_mixer(x)
        x = x.view(batch, self.num_patches, -1)

        # Expand the seq to match vocab
        x = self.seq_expansion(x)
        x = x.view(batch, self.seq_len, -1)

        return x


class AudioToLangEmbedding(nn.Module):
    """AudioToLangEmbedding

    Args:
        audio_seq_len (int): Length of the audio sequence
        seqlen (int): Length of the sequence
        dim (int): Embedding dimension

    Example:
        >>> import torch
        >>> from geminix import AudioToLangEmbedding
        >>> model = AudioToLangEmbedding(
        ...     audio_seq_len=32000,
        ...     seqlen=512,
        ...     dim=512
        ... )
        >>> x = torch.randn(1, 32000)
        >>> y = model(x)
        >>> y.shape
        torch.Size([1, 512, 512])
    """

    def __init__(self, audio_seq_len, seqlen, dim):
        super(AudioToLangEmbedding, self).__init__()
        self.audio_seq_len = audio_seq_len
        self.seqlen = seqlen
        self.dim = dim
        # Initialize a linear layer to project the 2D audio input to the desired 3D shape
        self.projection = nn.Linear(audio_seq_len, seqlen * dim)

    def forward(self, x):
        """Forward pass

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x shape: [batch, audio_seq_len] - 2D input
        batch, audio_seq_len = x.shape
        device = x.device

        # Project the audio tensor to match the seqlen and dim
        x = self.projection(x)  # x shape: [batch, seqlen * dim]

        # Reshape to the target shape: [batch, seqlen, dim]
        x = rearrange(x, "b (s d) -> b s d", s=self.seqlen, d=self.dim)

        return x


# # Example usage
# audio_seq_len = 32000  # Input audio sequence length
# seqlen = 512  # Sequence length to align with the language transformer
# dim = 512  # Embedding dimension

# model = AudioToLangEmbedding(audio_seq_len, seqlen, dim)
# audio_input = torch.randn(1, audio_seq_len)  # Example input tensor
# output = model(audio_input)

# print("Output shape:", output.shape)  # Should be [1, 512, 512]
