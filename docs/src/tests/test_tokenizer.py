import unittest

import pytest

from gemini_torch.tokenizer import (
    MultimodalSentencePieceTokenizer,
    SentencePieceProcessor,
)


# Fixture for tokenizer initialization
@pytest.fixture
def tokenizer():
    return MultimodalSentencePieceTokenizer()


# Test decoding of various token sequences
@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([1, 2, 3], "abc"),  # Replace with actual expected output
        ([4, 5, 6], "def"),  # Replace with actual expected output
        ([7, 8, 9], "ghi"),  # Replace with actual expected output
        # Add more test cases as needed
    ],
)
def test_decode(tokens, expected, tokenizer):
    assert tokenizer.decode(tokens) == expected


# Test decoding of sequences with modality tokens
@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([1, 2, 3, 100, 101], "abc"),  # Assuming 100 and 101 are modality tokens
        ([4, 5, 6, 102, 103], "def"),  # Assuming 102 and 103 are modality tokens
        ([7, 8, 9, 104, 105], "ghi"),  # Assuming 104 and 105 are modality tokens
        # Add more test cases as needed
    ],
)
def test_decode_with_modality_tokens(tokens, expected, tokenizer):
    assert tokenizer.decode(tokens) == expected


# Test decoding of empty sequence
def test_decode_empty_sequence(tokenizer):
    assert tokenizer.decode([]) == ""


# Test decoding of sequence with invalid tokens
def test_decode_invalid_tokens(tokenizer):
    with pytest.raises(Exception):  # Replace with the specific expected exception
        tokenizer.decode([999999])  # Assuming 999999 is an invalid token


# Test encoding of various inputs
@pytest.mark.parametrize(
    "input, expected",
    [
        ("abc", [1, 2, 3]),  # Replace with actual expected output
        ("def", [4, 5, 6]),  # Replace with actual expected output
        ("ghi", [7, 8, 9]),  # Replace with actual expected output
        # Add more test cases as needed
    ],
)
def test_encode(input, expected, tokenizer):
    assert tokenizer.encode(input) == expected


# Test encoding of empty string
def test_encode_empty_string(tokenizer):
    assert tokenizer.encode("") == []


# Test encoding of string with special characters
@pytest.mark.parametrize(
    "input, expected",
    [
        ("abc!", [1, 2, 3, 100]),  # Assuming 100 is the token for "!"
        ("def?", [4, 5, 6, 101]),  # Assuming 101 is the token for "?"
        ("ghi#", [7, 8, 9, 102]),  # Assuming 102 is the token for "#"
        # Add more test cases as needed
    ],
)
def test_encode_special_characters(input, expected, tokenizer):
    assert tokenizer.encode(input) == expected


# Test encoding of string of different lengths
@pytest.mark.parametrize(
    "input, expected",
    [
        ("a", [1]),  # Replace with actual expected output
        ("ab", [1, 2]),  # Replace with actual expected output
        ("abc", [1, 2, 3]),  # Replace with actual expected output
        ("abcd", [1, 2, 3, 4]),  # Replace with actual expected output
        ("abcde", [1, 2, 3, 4, 5]),  # Replace with actual expected output
        # Add more test cases as needed
    ],
)
def test_encode_different_lengths(input, expected, tokenizer):
    assert tokenizer.encode(input) == expected


class TestMultimodalSentencePieceTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MultimodalSentencePieceTokenizer(
            tokenizer_name="hf-internal-testing/llama-tokenizer"
        )

    # Tests for initialization
    def test_init_with_model_path(self):
        model_path = "path/to/tokenizer.model"
        tokenizer = MultimodalSentencePieceTokenizer(model_path=model_path)
        self.assertIsInstance(tokenizer.sp_model, SentencePieceProcessor)

    def test_init_with_tokenizer_name(self):
        tokenizer_name = "hf-internal-testing/llama-tokenizer"
        tokenizer = MultimodalSentencePieceTokenizer(tokenizer_name=tokenizer_name)
        self.assertIsInstance(tokenizer.sp_model, SentencePieceProcessor)

    def test_init_raises_error_without_model_or_name(self):
        with self.assertRaises(ValueError):
            MultimodalSentencePieceTokenizer()

    # Tests for model download
    def test_download_tokenizer(self):
        model_path = "data/tokenizer.model"
        downloaded_model_path = MultimodalSentencePieceTokenizer.download_tokenizer(
            "hf-internal-testing/llama-tokenizer"
        )
        self.assertEqual(model_path, downloaded_model_path)

    # Tests for encode
    def test_encode_with_text(self):
        encoded_text = self.tokenizer.encode("This is text.")
        self.assertIsInstance(encoded_text, list)
        self.assertEqual(len(encoded_text), 6)  # Assuming 6 tokens in the encoded text

    def test_encode_with_audio(self):
        encoded_audio = self.tokenizer.encode("Audio description", modality="audio")
        self.assertIsInstance(encoded_audio, list)
        self.assertIn(
            self.tokenizer.modality_tokens["audio"][0], encoded_audio
        )  # Checking for modality start token
        self.assertIn(
            self.tokenizer.modality_tokens["audio"][1], encoded_audio
        )  # Checking for modality end token

    def test_encode_with_image(self):
        encoded_image = self.tokenizer.encode("Image description", modality="image")
        self.assertIsInstance(encoded_image, list)
        self.assertIn(
            self.tokenizer.modality_tokens["image"][0], encoded_image
        )  # Checking for modality start token
        self.assertIn(
            self.tokenizer.modality_tokens["image"][1], encoded_image
        )  # Checking for modality end token

    def test_encode_with_bos_and_eos(self):
        encoded_text = self.tokenizer.encode("This is text.", bos=True, eos=True)
        self.assertEqual(encoded_text[0], self.tokenizer.bos_id)  # Checking BOS token
        self.assertEqual(encoded_text[-1], self.tokenizer.eos_id)  # Checking EOS token

    def test_encode_with_custom_bos_and_eos(self):
        custom_bos_id = 100
        custom_eos_id = 200
        encoded_text = self.tokenizer.encode(
            "This is text.", bos=custom_bos_id, eos=custom_eos_id
        )
        self.assertEqual(encoded_text[0], custom_bos_id)  # Checking custom BOS token
        self.assertEqual(encoded_text[-1], custom_eos_id)  # Checking custom EOS token

    # Tests for decode
    def test_decode_text(self):
        encoded_text = self.tokenizer.encode("This is text.")
        decoded_text = self.tokenizer.decode(encoded_text)
        self.assertEqual(decoded_text, "This is text.")

    def test_decode_audio(self):
        encoded_audio = self.tokenizer.encode("Audio description", modality="audio")
        decoded_audio = self.tokenizer.decode(encoded_audio)
        self.assertEqual(decoded_audio, "Audio description")  # Ignoring modality tokens

        # Tests for decode with different tokens

    def test_decode_with_unknown_tokens(self):
        encoded_text = self.tokenizer.encode("This is text. ")
        decoded_text = self.tokenizer.decode(encoded_text)
        self.assertEqual(
            decoded_text, "This is text. "
        )  # Unknown token remains unchanged

    # Tests for exception handling
    def test_encode_raises_error_with_invalid_modality(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode("Text description", modality="invalid_modality")

    # Tests for integration with other libraries
    def test_encode_decodes_with_transformers_tokenizer(self):
        from transformers import T5Tokenizer

        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        encoded_text_t5 = t5_tokenizer.encode("This is text.")
        encoded_text_sp = self.tokenizer.encode("This is text.")
        decoded_text_t5 = t5_tokenizer.decode(encoded_text_sp)
        self.assertEqual(
            encoded_text_sp, encoded_text_t5
        )  # Encoding should be consistent
        self.assertEqual(
            decoded_text_t5, "This is text."
        )  # Decoding should be consistent

    # Tests for coverage
    def test_model_attributes(self):
        self.assertIsInstance(self.tokenizer.sp_model, SentencePieceProcessor)
        self.assertIsInstance(self.tokenizer.n_words, int)
        self.assertIsInstance(self.tokenizer.bos_id, int)
        self.assertIsInstance(self.tokenizer.eos_id, int)
        self.assertIsInstance(self.tokenizer.pad_id, int)
        self.assertIsInstance(self.tokenizer.modality_tokens, dict)

    # Additional tests
    def test_encode_decodes_with_different_lengths(self):
        encoded_text_long = self.tokenizer.encode("This is a long text.")
        encoded_text_short = self.tokenizer.encode("This is short.")
        decoded_text_long = self.tokenizer.decode(encoded_text_long)
        decoded_text_short = self.tokenizer.decode(encoded_text_short)
        self.assertEqual(
            len(encoded_text_long), len(encoded_text_short)
        )  # Output length independent of input length
        self.assertEqual(decoded_text_long, "This is a long text.")
        self.assertEqual(decoded_text_short, "This is short.")

    def test_encode_decodes_with_special_characters(self):
        text_with_special_chars = "This text has #@! special characters."
        encoded_text = self.tokenizer.encode(text_with_special_chars)
        decoded_text = self.tokenizer.decode(encoded_text)
        self.assertEqual(decoded_text, text_with_special_chars)

    def test_tokenizer_is_pickleable(self):
        import pickle

        pickled_tokenizer = pickle.dumps(self.tokenizer)
        unpickled_tokenizer = pickle.loads(pickled_tokenizer)
        self.assertEqual(self.tokenizer.sp_model, unpickled_tokenizer.sp_model)
