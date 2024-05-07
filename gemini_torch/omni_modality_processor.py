import torch
from zeta.nn import audio_to_text, img_to_text, video_to_text
from torch import nn


def handle_omni_modality_processor(
    self,
    text: torch.Tensor = None,
    img: torch.Tensor = None,
    audio: torch.Tensor = None,
    video: torch.Tensor = None,
    model: nn.Module = None,
    post_modal_transform_norm: bool = False,
    *args,
    **kwargs,
):
    """
    Forward pass of the Gemini model.

    Args:
    - text: Text tensor
    - img: Image tensor
    - audio: Audio tensor
    - video: Video tensor
    - *args: Additional positional arguments
    - **kwargs: Additional keyword arguments

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

        if post_modal_transform_norm:
            img = nn.LayerNorm(img)

    if audio is not None:
        # Audio dimensions
        audio_b, audio_seq_len = audio.shape

        audio = audio_to_text(audio, self.max_seq_len, self.dim, True)

        if post_modal_transform_norm:
            audio = nn.LayerNorm(audio)

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
        fused = nn.LayerNorm(fused)

    return model(text, context=fused, *args, **kwargs)
