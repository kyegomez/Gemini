import unittest
import copy
import pytest
import torch

from gemini_torch.model import Gemini


# Fixture for model initialization
@pytest.fixture
def gemini_model():
    return Gemini()


# Test for default parameter initialization
def test_init_default_params(gemini_model):
    assert isinstance(gemini_model, Gemini), "Model is not an instance of Gemini"


# Test for forward pass with only text input
def test_forward_text_input_only(gemini_model):
    text_input = torch.randn(1, 50432, 2560)  # Adjust dimensions as necessary
    output = gemini_model(text=text_input)
    assert output is not None, "Output should not be None"
    assert output.shape == (1, 50432, 2560), f"Unexpected output shape: {output.shape}"


# Test for invalid text input shape
def test_invalid_text_input_shape(gemini_model):
    with pytest.raises(
        Exception
    ):  # Replace Exception with the specific expected exception
        invalid_text_input = torch.randn(1, 1, 1)  # Deliberately wrong shape
        _ = gemini_model(text=invalid_text_input)


class TestGemini(unittest.TestCase):
    def setUp(self):
        self.gemini = Gemini()
        self.text = torch.randn(
            1, 8192, 2560
        )  # batch size = 1, seq_len = 8192, dim = 2560
        self.img = torch.randn(
            1, 3, 256, 256
        )  # batch size = 1, channels = 3, height = 256, width = 256

    def test_initialization(self):
        self.assertIsInstance(self.gemini, Gemini)

    def test_forward_with_img(self):
        output = self.gemini(self.text, self.img)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape, (1, 8192, 2560)
        )  # batch size = 1, seq_len = 8192, dim = 2560

    def test_forward_without_img(self):
        output = self.gemini(self.text)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape, (1, 8192, 2560)
        )  # batch size = 1, seq_len = 8192, dim = 2560


def test_forward_img_input_only(gemini_model):
    img_input = torch.randn(1, 3, 64, 64)  # Assuming 64x64 is the appropriate size
    output = gemini_model(img=img_input)
    assert output is not None, "Output should not be None"
    # Add more assertions to verify output shape and other characteristics


def test_forward_both_inputs(gemini_model):
    text_input = torch.randn(1, 50432, 2560)
    img_input = torch.randn(1, 3, 64, 64)
    output = gemini_model(text=text_input, img=img_input)
    assert output is not None, "Output should not be None"
    # Add more assertions as needed


def test_model_with_max_seq_len(gemini_model):
    text_input = torch.randn(1, 8192, 2560)  # Assuming 8192 is the max sequence length
    output = gemini_model(text=text_input)
    assert (
        output.shape[1] == 8192
    ), "Output sequence length does not match max sequence length"


def test_forward_output_values_range(gemini_model):
    text_input = torch.randn(1, 50432, 2560)
    output = gemini_model(text=text_input)
    assert (
        output.max() <= 1 and output.min() >= -1
    ), "Output values are out of expected range [-1, 1]"


def test_model_with_high_dimension_input(gemini_model):
    text_input = torch.randn(1, 50432, 3000)  # Higher dimension than usual
    output = gemini_model(text=text_input)
    assert output is not None, "Model failed to process high dimension input"


def test_model_performance_with_typical_data(gemini_model):
    text_input = torch.randn(1, 50432, 2560)
    img_input = torch.randn(1, 3, 64, 64)
    output = gemini_model(text=text_input, img=img_input)
    assert output is not None, "Model failed with typical data"
    # Add more assertions as needed


def test_robustness_to_random_dropout(gemini_model):
    text_input = torch.randn(1, 50432, 2560) * torch.bernoulli(
        0.5 * torch.ones(1, 50432, 2560)
    )
    output = gemini_model(text=text_input)
    assert output is not None, "Model is not robust to random dropout"


def test_model_serialization(gemini_model):
    torch.save(gemini_model.state_dict(), "gemini_model.pth")
    deserialized_model = Gemini()
    deserialized_model.load_state_dict(torch.load("gemini_model.pth"))
    assert isinstance(deserialized_model, Gemini), "Deserialization failed"


def test_model_copy(gemini_model):
    model_copy = copy.deepcopy(gemini_model)
    assert isinstance(model_copy, Gemini), "Model copy is not an instance of Gemini"


def test_forward_pass_error_handling(gemini_model):
    with pytest.raises(ValueError):  # Replace ValueError with the expected error type
        invalid_input = torch.randn(1, 2, 3)  # Deliberately incorrect shape
        _ = gemini_model(text=invalid_input)
