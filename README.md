# gpt2-vision-language


<p align="center">
  <img src="images/Generation_1.jpg" width="100%" />
</p>

<p align="center">
    <img src="images/grid_3x3_BLIP.png" width="60%" />
  </p>

## Introduction

I present in this repository an independent project that I developed in parallel with my IASD master coursework.

This projects aims to:

1. **Build a GPT-2 decoder entirely from scratch** and pre-train it on a large corpus.  
2. **Fine-tune it for image captioning**, connecting a frozen vision encoder (CLIP ViT-L/14) to my frozen GPT-2 through different lightweight multimodal bridges.

I wanted to explore large-scale pre-training, understand the challenges of multimodal alignment, and experiment with vision representations. I implemented everything myself: model components, architecture blocks, and optimization. I experimented a more exploratory part in image-captioning task with a literature review of multimodal language models and the choice of three architectures few-resource-friendly (illustrated below): **linear projection**, **cross-attention**, and **Q-Former blocks inspired by BLIP-2** [Li et al., 2023].

All experiments were run on **a single NVIDIA RTX A5000 GPU**.

I obtained the following results:  
— Built a **124M-parameter GPT-2 decoder**, following Andrej Karpathy’s 10-lecture series, trained it on **FineWeb-Edu (10B tokens)**, and reached **30% HellaSwag accuracy** after ~2 days on my GPU.  
— Fine-tuned the model on **COCO 2017** for image captioning; after  **3 hours of training (1 epoch)** for each architecture, captions were already coherent with meaningful semantic content.  
More details about the architectures, results, and method are provided below.

---

## High-Level Overview

### **Pre-training**
- Implemented a **124M-parameter GPT-2 decoder** from scratch, inspired by Andrej Karpathy’s 10-lecture series.  
- Trained on **FineWeb-Edu (~10B tokens)**.  
- Used **FlashAttention** [Dao et al., 2022], **mixed precision**, and **gradient accumulation** (GPT-3-style effective batch size) [Brown et al., 2020].  
- Reached **≈30% accuracy on HellaSwag** after ~2 days of training on a single GPU.


### **Image captioning (COCO 2017)**
- Imported a **frozen CLIP ViT-L/14** encoder [Radford et al., 2021].  
- Kept both CLIP and the built GPT-2 **frozen**, training only small bridge modules.  
- Reduced CLIP’s 257 tokens (1 [CLS] + 256 patches) to a compact representation via **average pooling**.  
- Explored three lightweight architectures:
  1. **Linear projection**
  2. **Cross-attention layers**
  3. **Q-Former-style blocks inspired by BLIP-2** [Li et al., 2023]
- Fine-tuned on **MS-COCO 2017** [Chen et al., 2015].  
- After **~3 hours**, the model already produced coherent and semantically aligned captions.

---

## Part 1 — GPT-2 From Scratch

Following Karpathy’s 10-lecture series "Zero to Hero" that can be found with https://karpathy.ai/, I reconstructed a 124M-parameter GPT-2 decoder entirely by hand and trained it efficiently using FlashAttention [Dao et al., 2022], mixed precision, and GPT-3-inspired hyperparameters [Brown et al., 2020] with gradient accumulation.
After two days of training on FineWeb-Edu (10B tokens), the model achieved 30% HellaSwag accuracy.  

### Architecture  
The architecture is the same as in the GPT-2 paper [Radford et al., 2019].

<p align="center">
  <img src="images/GPT-2.png" width="20%" />
</p>

### Hyperparameters  
I reuse most architectural hyperparameters from the GPT-2 paper [Radford et al., 2019], and adopt the optimization setup from the GPT-3 paper [Brown et al., 2020]:

- **Model size:** $12$ layers, $12$ attention heads, hidden size $768$ (GPT-2)  
- **Context length:** $1024$ tokens (GPT-2)  
- **Vocabulary:** Tiktoken tokenizer, $\approx 50\text{k}$ tokens  
- **Effective batch size:** $524{,}288$ tokens/step $= 16 \times 1024 \times 32$ (GPT-3)  
- **Optimizer:** AdamW with $\beta = (0.9,\ 0.95)$, weight decay $= 0.1$, gradient clipping $= 1$ (GPT-3)  
- **Learning-rate schedule:** cosine decay $(6\times10^{-4} \rightarrow 6\times10^{-5})$ (GPT-3), with 715 warmup steps

### Training curves  

<p align="center">
  <img src="images/val_loss.png" width="45%" />
  <img src="images/hellaswag_acc.png" width="45%" />
</p>

### Sample generations  

<p align="center">
  <img src="images/Generation_1.jpg" width="100%" />
</p>

---

## Part 2 — Multimodality

I imported **CLIP ViT-L/14** [Radford et al., 2021], kept both CLIP and the decoder GPT-2 **frozen**, and trained small bridging modules for image captioning.

For all three models, I reduced CLIP’s 257 embeddings (1 [CLS] + 256 patch embeddings), into 33 embeddings using **average pooling** for computational efficiency.

I then trained three architectures on **COCO 2017** [Chen et al., 2015] (118k train, 5k val, 5 captions/image).

---

## Part 2.1 — Linear Projection Bridge

A single learned linear layer maps the pooled CLIP visual tokens directly into the GPT-2 embedding space.

- **Architecture:**
Blocks highlighted in blue denote frozen parameters, whereas blocks in red correspond to trainable components.
  <p align="center">
    <img src="images/Linear_projection_architecture.png" width="60%" />
  </p>

- **Sample captions:**  

  <p align="center">
    <img src="images/grid_3x3_linearmodel.png" width="60%" />
  </p>

---

## Part 2.2 — Cross-Attention Bridge

Cross-attention layers are added inside the transformer blocks of the decoder, maintaining every parameters of other layers in the blocks frozen.

- **Architecture:**
Blocks highlighted in blue denote frozen parameters, whereas blocks in red correspond to trainable components.
  <p align="center">
    <img src="images/cross_attention_architecture.png" width="60%" />
  </p>

- **Sample captions:**  

  <p align="center">
    <img src="images/grid_3x3_crossattn.png" width="60%" />
  </p>

---

## Part 2.3 — Q-Former Bridge (Vision → Language)

A set of learnable queries attends to frozen CLIP features, producing a compact set of multimodal embeddings projected into GPT-2’s embedding space.

- **BLIP-2 reference architecture:** (from [Li et al., 2023])

  <p align="center">
    <img src="images/BLIP2.png" width="60%" />
  </p>

- **Architecture:**  
Blocks highlighted in blue denote frozen parameters, whereas blocks in red correspond to trainable components.
  <p align="center">
    <img src="images/Q_former_architecture.png" width="60%" />
  </p>

- **Sample captions:**  

  <p align="center">
    <img src="images/grid_3x3_BLIP.png" width="60%" />
  </p>

---

## Part 2.4 — Results

A single epoch (~3 hours) with AdamW, cosine learning rate decay, and batch size adapted through gradient accumulation was sufficient for all three bridging models to converge to stable validation loss curves.

### **Validation loss comparison**  

<p align="center">
  <img src="images/val_loss_comparison.png" width="60%" />
</p>


### **Captioning metrics (CIDEr, METEOR, etc.)**

<div align="center">

  
| Model             | METEOR ↑        | CIDEr ↑   |
|-------------------|-----------------|-----------|
| Cross-Attention   | 0.334 ± 0.153   | 0.321     |
| Linear Projection | 0.379 ± 0.139   | 0.419     |
| **Q-Former**      | **0.412 ± 0.146** | **0.598** |

</div>


---

## References

- Chen et al. (2015), *Microsoft COCO Captions*, arXiv:1504.00325.  
- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), arXiv:2103.00020.  
- Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135.  
- Li et al. (2023), *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, arXiv:2301.12597.  
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3), arXiv:2005.14165.  
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2), OpenAI Technical Report.  
- Vedantam et al. (2015), *CIDEr: Consensus-based Image Description Evaluation*, arXiv:1411.5726.

