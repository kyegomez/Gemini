import torch
import pytest
from gemini_torch.utils import AudioToLangEmbedding


@pytest.fixture
def audio_embedding():
    audio_seq_len = 32000
    seqlen = 512
    dim = 512
    return AudioToLangEmbedding(audio_seq_len, seqlen, dim)


def test_forward_pass(audio_embedding):
    # Test the forward pass with a random input
    batch_size = 2
    input_audio = torch.randn(batch_size, audio_embedding.audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (batch_size, audio_embedding.seqlen, audio_embedding.dim)


def test_device_placement(audio_embedding):
    # Test if the model and input/output tensors are on the same device
    input_audio = torch.randn(1, audio_embedding.audio_seq_len)
    output = audio_embedding(input_audio)
    assert input_audio.device == output.device
    assert input_audio.device == audio_embedding.projection.weight.device


def test_output_shape(audio_embedding):
    # Test if the output shape matches the expected shape
    input_audio = torch.randn(1, audio_embedding.audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (1, audio_embedding.seqlen, audio_embedding.dim)


def test_batch_processing(audio_embedding):
    # Test batch processing by passing a batch of input tensors
    batch_size = 4
    input_audio = torch.randn(batch_size, audio_embedding.audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (batch_size, audio_embedding.seqlen, audio_embedding.dim)


def test_zero_input(audio_embedding):
    # Test with zero input
    input_audio = torch.zeros(1, audio_embedding.audio_seq_len)
    output = audio_embedding(input_audio)
    assert torch.all(output == 0)


def test_negative_input(audio_embedding):
    # Test with negative input values
    input_audio = torch.randn(1, audio_embedding.audio_seq_len) - 2.0
    output = audio_embedding(input_audio)
    assert torch.all(output < 0)


def test_large_input(audio_embedding):
    # Test with large input values
    input_audio = torch.randn(1, audio_embedding.audio_seq_len) * 100.0
    output = audio_embedding(input_audio)
    assert torch.all(output > 0)


def test_input_shape_mismatch(audio_embedding):
    # Test if an error is raised for an input shape mismatch
    with pytest.raises(torch.nn.modules.module.ModuleAttributeError):
        input_audio = torch.randn(1, audio_embedding.audio_seq_len + 1)
        audio_embedding(input_audio)


def test_output_device(audio_embedding):
    # Test if the output device matches the expected device
    input_audio = torch.randn(1, audio_embedding.audio_seq_len).to("cuda")
    audio_embedding.to("cuda")
    output = audio_embedding(input_audio)
    assert output.device == torch.device("cuda")


def test_large_batch_size(audio_embedding):
    # Test with a large batch size
    batch_size = 1024
    input_audio = torch.randn(batch_size, audio_embedding.audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (batch_size, audio_embedding.seqlen, audio_embedding.dim)


def test_small_batch_size(audio_embedding):
    # Test with a small batch size (1)
    input_audio = torch.randn(1, audio_embedding.audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (1, audio_embedding.seqlen, audio_embedding.dim)


def test_audio_seq_len_equal_seqlen(audio_embedding):
    # Test when audio_seq_len is equal to seqlen
    audio_seq_len = seqlen = 512
    dim = 512
    audio_embedding = AudioToLangEmbedding(audio_seq_len, seqlen, dim)
    input_audio = torch.randn(1, audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (1, seqlen, dim)


def test_audio_seq_len_less_than_seqlen(audio_embedding):
    # Test when audio_seq_len is less than seqlen
    audio_seq_len = 256
    seqlen = 512
    dim = 512
    audio_embedding = AudioToLangEmbedding(audio_seq_len, seqlen, dim)
    input_audio = torch.randn(1, audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (1, seqlen, dim)


def test_audio_seq_len_greater_than_seqlen(audio_embedding):
    # Test when audio_seq_len is greater than seqlen
    audio_seq_len = 1024
    seqlen = 512
    dim = 512
    audio_embedding = AudioToLangEmbedding(audio_seq_len, seqlen, dim)
    input_audio = torch.randn(1, audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (1, seqlen, dim)


def test_dim_less_than_seqlen(audio_embedding):
    # Test when dim is less than seqlen
    audio_seq_len = 32000
    seqlen = 512
    dim = 256
    audio_embedding = AudioToLangEmbedding(audio_seq_len, seqlen, dim)
    input_audio = torch.randn(1, audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (1, seqlen, dim)


def test_dim_greater_than_seqlen(audio_embedding):
    # Test when dim is greater than seqlen
    audio_seq_len = 32000
    seqlen = 512
    dim = 1024
    audio_embedding = AudioToLangEmbedding(audio_seq_len, seqlen, dim)
    input_audio = torch.randn(1, audio_seq_len)
    output = audio_embedding(input_audio)
    assert output.shape == (1, seqlen, dim)
