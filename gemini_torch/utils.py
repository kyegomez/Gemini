from einops import rearrange, reduce
from torch import nn


class ImageToTextEmbeddings(nn.Module):
    """
    Converts images into text tokens using patch-based embedding.

    Args:
        patch_size (int): The size of each patch in the image.
        dim (int): The dimension of the embedding for each patch.
        seq_len (int): The desired sequence length of the text tokens.

    Returns:
        torch.Tensor: The text tokens representing the input images.

    """

    def __init__(self, patch_size: int, dim: int, seq_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.seq_len = seq_len
        self.projection = nn.Linear(patch_size * patch_size * 3, dim)

    def forward(self, images):
        # Input images are assumed to be in the shape (batch_size, channels, height, width)
        batch_size, _, height, width = images.shape

        # Ensure that the image dimensions are divisible by the patch size
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size"

        # Rearrange the images into patches using einops
        patches = rearrange(
            images,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        # Project the patches into the embedding dimension
        embeddings = self.projection(patches)

        # Reshape the embeddings into the shape (batch_size, seq_len, dim)
        seq_len = (height // self.patch_size) * (width // self.patch_size)
        text_tokens = rearrange(embeddings, "b (h w) e -> b h w e", h=seq_len, w=1)
        text_tokens = reduce(text_tokens, "b h w e -> b (w e h)", "mean")
        seq_proj = nn.Linear(seq_len, self.seq_len)
        text_tokens = seq_proj(text_tokens)

        return text_tokens


# x = torch.randn(1, 3, 64, 64)
# model = ImageToTextEmbeddings(patch_size=8, dim=512, seq_len=128)
# y = model(x)
# print(y.shape)  # Should be [1, 64, 512]


# x = torch.randn(1, 3, 64, 64)
# model = ImageToTextEmbeddings(patch_size=8, dim=512, seq_len=128)
# y = model(x)
# print(y.shape)  # Should be [1, 64, 512]


class AudioToEmbeddings(nn.Module):
    """AudioToEmbeddings

    Args:
        audio_seq_len (int): Length of the audio sequence
        seqlen (int): Length of the sequence
        dim (int): Embedding dimension

    Example:
        >>> import torch
        >>> from geminix import AudioToEmbeddings
        >>> model = AudioToEmbeddings(
        ...     audio_seq_len=32000,
        ...     seqlen=512,
        ...     dim=512
        ... )
        >>> x = torch.randn(1, 32000)
        >>> y = model(x)
        >>> y.shape
        torch.Size([1, 512, 512])
    """

    def __init__(self, audio_seq_len: int, seqlen: int, dim: int):
        super(AudioToEmbeddings, self).__init__()
        self.audio_seq_len = audio_seq_len
        self.seqlen = seqlen
        self.dim = dim
        # Initialize a linear layer to project the 2D audio input to the desired 3D shape
        self.projection = nn.Linear(audio_seq_len, dim)

    def forward(self, x):
        """Forward pass

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x shape: [batch, audio_seq_len] - 2D input
        batch, audio_seq_len = x.shape

        # Project the audio tensor to match the seqlen and dim
        x = self.projection(x)  # x shape: [batch, seqlen * dim]

        # Reshape to the target shape: [batch, seqlen, dim]
        x = rearrange(x, "b (s d) -> b s d", s=self.seqlen, d=self.dim)

        return x


# x = torch.randn(1, 32000)
# model = AudioToEmbeddings(
#     audio_seq_len=32000,
#     seqlen=512,
#     dim=512
# )
# y = model(x)
# print(y.shape)  # Should be [1, 512, 512]
