# gpt2-vision-language


<p align="center">
  <img src="images/Generation_1.png" width="100%" />
</p>

<p align="center">
    <img src="images/grid_3x3_BLIP.png" width="60%" />
  </p>

## Introduction

We present in this repository an independent project that we developed in parallel with our IASD master coursework.

This projects aims to:

1. **Build a GPT-2 decoder entirely from scratch** and pre-train it on a large corpus.  
2. **Fine-tune it for image captioning**, connecting a frozen vision encoder (CLIP ViT-L/14) to our frozen GPT-2 through different lightweight multimodal bridges.

We wanted to explore large-scale pre-training, understand the challenges of multimodal alignment, and experiment with vision representations. We implemented everything ourselves: model components, architecture blocks, and optimization. We experimented a more exploratory part in image-captioning task with a literature review of multimodal language models and the choice of three architectures few-resource-friendly (illustrated below): **linear projection**, **cross-attention**, and **Q-Former blocks inspired by BLIP-2**.

All experiments were run on **a single NVIDIA RTX A5000 GPU**.

We obtained the following results:  
— Built a **124M-parameter GPT-2 decoder**, following Andrej Karpathy’s 10-lecture series, trained it on **FineWeb-Edu (10B tokens)**, and reached **31% HellaSwag accuracy** after ~2 days on our GPU.  
— Fine-tuned the model on **COCO 2017** for image captioning; after  **3 hours of training (1 epoch)** for each architecture, captions were already coherent with meaningful semantic content.  
More details are provided below.

---

## High-Level Overview

### **Pre-training**
- Implemented a **124M-parameter GPT-2 decoder** from scratch, inspired by Karpathy’s 10-lecture series.  
- Trained on **FineWeb-Edu (~10B tokens)**.  
- Used **FlashAttention**, **mixed precision**, and **gradient accumulation** (GPT-3-style effective batch size).  
- Reached **≈31% accuracy on HellaSwag** after ~2 days of training on a single GPU.


### **Image captioning (COCO 2017)**
- Imported a **frozen CLIP ViT-L/14** encoder.  
- Kept both CLIP and GPT-2 **frozen**, training only small bridge modules.  
- Reduced CLIP’s 257 tokens (1 [CLS] + 256 patches) to a compact representation via **average pooling**.  
- Explored three lightweight architectures:
  1. **Linear projection**
  2. **Cross-attention layers**
  3. **Q-Former-style blocks inspired by BLIP-2**
- Fine-tuned on **MS-COCO 2017** (118k train, 5k val, 5 captions/image).  
- After **~3 hours**, the model already produced coherent and semantically aligned captions.

---

## Part 1 — GPT-2 From Scratch

Following Karpathy’s 10-lecture series "Zero to Hero" that can be found with https://karpathy.ai/, we reconstructed a 124M-parameter GPT-2 decoder entirely by hand and trained it efficiently using FlashAttention, mixed precision, and GPT-3-inspired hyperparameters with gradient accumulation.
After two days of training on FineWeb-Edu (10B tokens), the model achieved 31% HellaSwag accuracy.  

### Architecture  

<p align="center">
  <img src="images/GPT-2.png" width="20%" />
</p>


### Training curves  

<p align="center">
  <img src="images/val_loss.png" width="45%" />
  <img src="images/hellaswag_acc.png" width="45%" />
</p>

### Sample generations  

<p align="center">
  <img src="images/Generation_1.png" width="100%" />
</p>

---

## Part 2 — Multimodality

We imported **CLIP ViT-L/14**, kept both CLIP and the decoder GPT-2 **frozen**, and trained small bridging modules for image captioning.

For all three models, we reduced CLIP’s 257 embeddings (1 [CLS] + 256 patch embeddings), into 33 embeddings using **average pooling** for computational efficiency.

We then trained three architectures on **COCO 2017** (118k train, 5k val, 5 captions/image).

---

## Part 2.1 — Linear Projection Bridge

A single learned linear layer maps the pooled CLIP visual tokens directly into the GPT-2 embedding space.

- **Architecture:**
- Blocks with frozen parameters are in blue, and trainable parameters are in red.
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

- **BLIP-2 reference:**  

  <p align="center">
    <img src="images/BLIP2.png" width="60%" />
  </p>

- **Architecture:**  

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


## Part 3 — Installation and training

## Part 3.1 — Install

## Part 3.2 — Train

## Part 3.3 — Generate


## Acknowledgements


## Contributors


This project has been conducted by Théophile Laurent, Gabriel Lenain and Lucas Mebille. (liens Github)





---
