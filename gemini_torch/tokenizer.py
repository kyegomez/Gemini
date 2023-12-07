import os
from logging import getLogger
from typing import List, Optional

import requests
from sentencepiece import SentencePieceProcessor

logger = getLogger()

class MultimodalSentencePieceTokenizer:
    """
    A tokenizer that extends the SentencePieceTokenizer for multi-modality inputs.
    It includes special tokens for different modalities like text, audio, image, etc.
    """

    def __init__(self, model_path: str = "tokenizer.model"):
        # Ensure the model file exists
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # Initialize token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        # Initialize special token IDs for modalities
        self.img_token_id: int = self.sp_model.piece_to_id("<img>") or -1
        self.audio_token_id: int = self.sp_model.piece_to_id("<audio>") or -1

        logger.info(
            f"#words: {self.n_words}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}, "
            f"Image Token ID: {self.img_token_id}, Audio Token ID: {self.audio_token_id}"
        )
        
    def download_model(self, model_path: str):
        # download the model from the model path
        request = requests.get(model_path, stream=True)
        with open("tokenizer.model", "wb") as file:
            for chunk in request.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        logger.info(f"Downloaded SentencePiece model from {model_path}")
        

    def encode(self, s: str, modality: str, bos: bool = True, eos: bool = True) -> List[int]:
        assert isinstance(s, str)
        tokens = self.sp_model.encode(s)

        # Prepend special modality token if available
        if modality == 'image' and self.img_token_id != -1:
            tokens = [self.img_token_id] + tokens
        elif modality == 'audio' and self.audio_token_id != -1:
            tokens = [self.audio_token_id] + tokens

        # Add BOS/EOS tokens if required
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        # Remove special modality tokens before decoding
        tokens = [t for t in tokens if t not in (self.img_token_id, self.audio_token_id)]
        return self.sp_model.decode(tokens)

# Example usage
tokenizer = MultimodalSentencePieceTokenizer()

# Encoding and decoding examples
encoded_text = tokenizer.encode("Example text", modality='text')
decoded_text = tokenizer.decode(encoded_text)

encoded_image = tokenizer.encode("Image description", modality='image')
decoded_image = tokenizer.decode(encoded_image)

print("Encoded text:", encoded_text)
print("Decoded text:", decoded_text)
print("Encoded image:", encoded_image)
print("Decoded image:", decoded_image)
