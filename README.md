[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Gemini

![gemini](gemini.png)

The open source implementation of Gemini, the model that will "eclipse ChatGPT", it seems to work by directly taking in all modalities all at once into a transformer with special decoders for text or img generation!

[Join the Agora discord channel to help with the implementation!](https://discord.gg/CMDpRxCV8g) and [Here is the project board:](https://github.com/users/kyegomez/projects/11/views/1)

The input sequences for Gemini consist of texts, audio, images, and videos. These inputs are transformed into tokens, which are then processed by a transformer. Subsequently, conditional decoding takes place to generate image outputs.

Interestingly, the architecture of Gemini bears resemblance to Fuyu's architecture but is expanded to encompass multiple modalities. Instead of utilizing a visual transformer (vit) encoder, Gemini simply feeds image embeddings directly into the transformer.

For Gemini, the token inputs will likely be indicated by special modality tokens such as [IMG], <img>, [AUDIO], or <audio>. Codi, a component of Gemini, also employs conditional generation and makes use of the tokenized outputs.

To implement this model effectively, I intend to initially focus on the image embeddings to ensure their smooth integration. Subsequently, I will proceed with incorporating audio embeddings and then video embeddings.

# Install
`pip3 install gemini-torch`


## Usage

### Gemini Transformer Usage
- Base transformer
- Multi Grouped Query Attn / flash attn
- rope
- alibi
- xpos
- qk norm
- no pos embeds
- kv cache

```python
import torch 
from gemini_torch import Gemini

# Initialize the model
model = Gemini(
    num_tokens=50432,
    max_seq_len=8192,
    dim=2560,
    depth=32,
    dim_head=128,
    heads=24,
    use_abs_pos_emb=False,
    alibi_pos_bias=True,
    alibi_num_heads=12,
    rotary_xpos=True,
    attn_flash=True,
    attn_kv_heads=2,
    qk_norm=True,
    attn_qk_norm=True,
    attn_qk_norm_dim_scale=True,
)

# Initialize the text random tokens
x = torch.randint(0, 50432, (1, 8192))

# Apply model to x
y = model(x)

# Print logits
print(y)
```
--------

### Multi-Modal with Imgs + Audio
- Img processing through a specially crafted module that takes in img -> patches it -> then reshapes to the shape of the text tensors, [B, seqlen, dim] -> align with text tokens

```python
import torch
from gemini_torch.model import Gemini

# Initialize model
model = Gemini(
    num_tokens=50432,
    max_seq_len=8192,
    dim=2560,
    depth=32,
    dim_head=128,
    heads=24,
    use_abs_pos_emb=False,
    alibi_pos_bias=True,
    alibi_num_heads=12,
    rotary_xpos=True,
    attn_flash=True,
    attn_kv_heads=2,
    qk_norm=True,
    attn_qk_norm=True,
    attn_qk_norm_dim_scale=True,
)

# Text shape: [batch, seq_len, dim]
text = torch.randint(0, 50432, (1, 8192))

# Img shape: [batch, channels, height, width]
img = torch.randn(1, 3, 256, 256)

# Audio shape: [batch, audio_seq_len, dim]
audio = torch.randn(1, 128)

# Apply model to text and img
y = model(text, img, audio)

# Output shape: [batch, seq_len, dim]
print(y.shape)


```
------



## Tokenizer
- Sentencepiece, tokenizer
- We're using the same tokenizer as LLAMA with special tokens denoting the beginning and end of the multi modality tokens.
- Does not fully process img, audio, or videos now we need help on that

```python
from gemini_torch.tokenizer import MultimodalSentencePieceTokenizer

# Example usage
tokenizer_name = "hf-internal-testing/llama-tokenizer"
tokenizer = MultimodalSentencePieceTokenizer(tokenizer_name=tokenizer_name)

# Encoding and decoding examples
encoded_audio = tokenizer.encode("Audio description", modality="audio")
decoded_audio = tokenizer.decode(encoded_audio)

print("Encoded audio:", encoded_audio)
print("Decoded audio:", decoded_audio)


```

### `ImgToTransformer`
- takes in img -> patches -> reshapes to [B, SEQLEN, Dim] to align with transformer
```python
import torch
from gemini_torch.utils import ImgToTransformer

# Example usage
num_patches = 16
patch_size = 16
transformer_dim = 512
img_channels = 3
seq_len = 50000
reduced_dim = 256  # Reduced dimension after dimensionality reduction

model = ImgToTransformer(
    num_patches, patch_size, transformer_dim, img_channels, seq_len, reduced_dim
)

# Dummy image input [BATCH, CHANNELS, HEIGHT, WIDTH]
dummy_img = torch.randn(1, 3, 64, 64)  # Batch size of 1, 64x64 RGB image

# Forward pass
seq_space_output = model(dummy_img)
print(seq_space_output.shape)  # Expected shape: [1, 50000, 256]


```

### `AudioToLangEmbedding`
- Transforms audio into the same shape as text tensors.

```python
import torch 
from gemini_torch.utils import AudioToLangEmbedding

# Example usage
audio_seq_len = 32000  # Input audio sequence length
seqlen = 512  # Sequence length to align with the language transformer
dim = 512  # Embedding dimension

model = AudioToLangEmbedding(audio_seq_len, seqlen, dim)
audio_input = torch.randn(1, audio_seq_len)  # Example input tensor
output = model(audio_input)

print("Output shape:", output.shape)  # Should be [1, 512, 512]

```


# References
* Combine Reinforcment learning with modular pretrained transformer, multi-modal capabilities, image, audio, 
* self improving mechanisms like robocat
* PPO? or MPO
* get good at backtracking and exploring alternative paths
* speculative decoding
* Algorithm of Thoughts
* RLHF
* [Gemini Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
* [Gemini Landing Page](https://deepmind.google/technologies/gemini/#introduction)


# Todo

- [ ] [Check out the project board for more todos](https://github.com/users/kyegomez/projects/11/views/1)


- [ ] Implement the img feature embedder and align imgs with text and pass into transformer: ```Gemini models are trained to accommodate textual input interleaved with a wide variety of audio and visual inputs, such as natural images, charts, screenshots, PDFs, and videos, and they can produce
text and image outputs (see Figure 2). The visual encoding of Gemini models is inspired by our own
foundational work on Flamingo (Alayrac et al., 2022), CoCa (Yu et al., 2022a), and PaLI (Chen et al.,
2022), with the important distinction that the models are multimodal from the beginning and can
natively output images using discrete image tokens (Ramesh et al., 2021; Yu et al., 2022b).```

- [ ] Implement the audio processing using USM by Google:```In addition, Gemini can directly ingest audio signals at
16kHz from Universal Speech Model (USM) (Zhang et al., 2023) features. This enables the model to
capture nuances that are typically lost when the audio is naively mapped to a text input (for example,
see audio understanding demo on the website).```


- [ ] Video Processing Technique: "
Video understanding is accomplished by encoding the video as a sequence of frames in the large
context window. Video frames or images can be interleaved naturally with text or audio as part of the
model input"

- [ ] Prompting Technique: ``` We find Gemini Ultra achieves highest
accuracy when used in combination with a chain-of-thought prompting approach (Wei et al., 2022)
that accounts for model uncertainty. The model produces a chain of thought with k samples, for
example 8 or 32. If there is a consensus above a preset threshold (selected based on the validation
split), it selects this answer, otherwise it reverts to a greedy sample based on maximum likelihood
choice without chain of thought. We refer the reader to appendix for a detailed breakdown of how
this approach compares with only chain-of-thought prompting or only greedy sampling.```



- [ ] Train a 1.8B + 3.25 Model: ```Nano-1 and Nano-2 model sizes are only 1.8B and 3.25B
parameters respectively. Despite their size, they show exceptionally strong performance on factuality,
i.e. retrieval-related tasks, and significant performance on reasoning, STEM, coding, multimodal and```
