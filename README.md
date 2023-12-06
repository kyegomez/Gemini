[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Gemini

![gemini](gemini.png)

The open source implementation of Gemini, the model that will "eclipse ChatGPT", it seems to work by directly taking in all modalities without an encoder for some kind which means that the encoding is built into the modal.

input sequences {texts, audio, imgs, video} -> [tokens] -> transformer -> conditional decoding for img gen

This architecture looks very similiar to Fuyu's architecture just extended to many modalities, where instead of an vit encoder you just pass in the img embeddings into the transformer.

The token inputs to gemini will most likely be denoted by special modality tokens `[IMG] or <img> or [AUDIO] or <audio>`

Codi also has conditional generation leverages the tokenized outputs.

To implement this, I plan to cover the img embedding first make sure that works well and then go onto the audio embeddings and then the video.






# References
* Combine Reinforcment learning with modular pretrained transformer, multi-modal capabilities, image, audio, 
* self improving mechanisms like robocat
* PPO? or MPO
* get good at backtracking and exploring alternative paths
* speculative decoding
* Algorithm of Thoughts
* RLHF
* ![Gemini Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
* ![Gemini Landing Page](https://deepmind.google/technologies/gemini/#introduction)


# Todo
- [ ] Implement the img feature embedder and align imgs with text and pass into transformer
