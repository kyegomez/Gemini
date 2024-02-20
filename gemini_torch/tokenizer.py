import os
from logging import getLogger
from typing import List, Optional

import requests
from sentencepiece import SentencePieceProcessor

logger = getLogger()

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}


class MultimodalSentencePieceTokenizer:
    """Multimodal SentencePiece tokenizer.

    Args:
        model_path (str, optional): Path to the SentencePiece model file. Defaults to None.
        tokenizer_name (str, optional): Name of the tokenizer to download. Defaults to None.

    Methods:
        encode(s: str, modality: str, bos: bool = True, eos: bool = True) -> List[int]: Encodes a string into a list of token IDs.
        decode(tokens: List[int]) -> str: Decodes a list of token IDs into a string.

    Examples:
        >>> tokenizer_name = "hf-internal-testing/llama-tokenizer"
        >>> tokenizer = MultimodalSentencePieceTokenizer(tokenizer_name=tokenizer_name)
        >>> encoded_audio = tokenizer.encode("Audio description", modality='audio')
        >>> decoded_audio = tokenizer.decode(encoded_audio)
        >>> print("Encoded audio:", encoded_audio)
        >>> print("Decoded audio:", decoded_audio)
    """

    def __init__(
        self, model_path: Optional[str] = None, tokenizer_name: Optional[str] = None
    ):
        if model_path:
            assert os.path.isfile(model_path), model_path
        elif tokenizer_name:
            model_path = self.download_tokenizer(tokenizer_name)
        else:
            raise ValueError("Either model_path or tokenizer_name must be provided.")

        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # Initialize token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        # Initialize special token IDs for modalities
        self.modality_tokens = {
            "image": (
                self.sp_model.piece_to_id("<img>"),
                self.sp_model.piece_to_id("</img>"),
            ),
            "audio": (
                self.sp_model.piece_to_id("<audio>"),
                self.sp_model.piece_to_id("</audio>"),
            ),
        }

    @staticmethod
    def download_tokenizer(tokenizer_name: str) -> str:
        """Downloads the SentencePiece model file from HuggingFace Hub.

        Args:
            tokenizer_name (str): _description_

        Raises:
            ValueError: _description_
            Exception: _description_

        Returns:
            str: _description_
        """
        if tokenizer_name not in PRETRAINED_VOCAB_FILES_MAP["vocab_file"]:
            raise ValueError(f"Tokenizer {tokenizer_name} is not available.")

        model_url = PRETRAINED_VOCAB_FILES_MAP["vocab_file"][tokenizer_name]
        model_path = os.path.join("data", "tokenizer.model")

        if not os.path.exists("data"):
            os.makedirs("data")

        # Downloading the tokenizer model file
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as file:
                file.write(response.content)
            logger.info(f"Downloaded SentencePiece model to {model_path}")
        else:
            raise Exception(f"Failed to download model from {model_url}")

        return model_path

    def encode(
        self, s: str, modality: str, bos: bool = True, eos: bool = True
    ) -> List[int]:
        """Encodes a string into a list of token IDs.

        Args:
            s (str): _description_
            modality (str): _description_
            bos (bool, optional): _description_. Defaults to True.
            eos (bool, optional): _description_. Defaults to True.

        Returns:
            List[int]: _description_
        """
        assert isinstance(s, str)
        tokens = self.sp_model.encode(s)

        # Prepend start and append end modality tokens if available
        modality_start_id, modality_end_id = self.modality_tokens.get(
            modality, (-1, -1)
        )
        if modality_start_id != -1 and modality_end_id != -1:
            tokens = [modality_start_id] + tokens + [modality_end_id]

        # Add BOS/EOS tokens if required
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """decodes a list of token IDs into a string.

        Args:
            tokens (List[int]): _description_

        Returns:
            str: _description_
        """
        # Remove modality tokens before decoding
        for start_id, end_id in self.modality_tokens.values():
            tokens = [t for t in tokens if t not in (start_id, end_id)]
        return self.sp_model.decode(tokens)
