from gemini_torch.tokenizer import MultimodalSentencePieceTokenizer

# Example usage
tokenizer_name = "hf-internal-testing/llama-tokenizer"
tokenizer = MultimodalSentencePieceTokenizer(tokenizer_name=tokenizer_name)

# Encoding and decoding examples
encoded_audio = tokenizer.encode("Audio description", modality="audio")
decoded_audio = tokenizer.decode(encoded_audio)

print("Encoded audio:", encoded_audio)
print("Decoded audio:", decoded_audio)
