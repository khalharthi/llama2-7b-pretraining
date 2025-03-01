# Llama2 7B Pre-training Scripts

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains the scripts for pre-training the Llama2 7B language model from scratch using the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset. The project focuses on efficient generation using KV (Key-Value) caching and periodic evaluation using the Hellaswag benchmark.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Folder Structure](#folder-structure)
- [Tokens and Batches per GPU](#tokens-and-batches-per-gpu)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Features
- **Llama2 7B Pre-training Scripts**: Contains the code to pre-train the Llama2 7B model from scratch using the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset.
  
- **KV Cache for Efficient Generation**: Implements KV caching during generation to reduce memory usage and improve inference speed.
  *Note: KV cache is typically used during inference, but here it is also used during pre-training when sampling from the model with the prompt "Hello, I'm a large language model, " to monitor progress.*
  
- **Periodic Generation During Training**: After every 250 steps, the script generates text using the prompt: `"Hello, I'm a large language model, "` and produces up to 40 tokens to monitor model progress.
  
- **Hellaswag Evaluation**: Evaluates the model using the Hellaswag benchmark every 250 steps and at the final step to measure commonsense reasoning capabilities.

---

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/khalharthi/llama2-7b-pretraining.git
   cd llama2-7b-pretraining
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   
3. Download and process the **FineWeb-Edu** dataset:
   ```bash
   python fineweb.py
   ```

   
4. Start pre-training:
   ```bash
   python llama2_training.py
   ```
---


## Evaluation
The model is evaluated on the Hellaswag benchmark every 250 steps and at the final step. The evaluation script calculates the accuracy of the model's predictions and logs the results.

---

## Folder Structure

* **llama2_training.py**: Contains the full code for pre-training the Llama2 7B model.

* **hellaswag.py**: Evaluates the model on the Hellaswag benchmark.

* **fineweb.py**: Installs fineweb-edu dataset.
  
* **requirements.txt**: Project's requirements (downloaded using ```pip -r install requirements.txt```).

---

## Tokens and Batches per GPU
The training configuration is as follows:

* **Batch Size per GPU**: B = 64
* **Sequence Length**: T = 4096
* **Number of GPUs**: ddp_world_size = 8 (using 8x A100 GPUs)
* **Total Tokens per GPU**: B * T = 262,144
* **Total Tokens per 1 Step**: B * T * 8 = 2,097,152
* **Total Training Tokens**: *FineWeb-Edu* dataset contains 9,853,989,344 tokens for training, and 100,000,000 for validation
* **number of steps per 1 epoch**: 9,853,989,344 / 2,097,152 ≈ 4,700 steps
* **Estimated Training Time per Epoch**: Each step takes **~30 seconds** (for 8x A100 GPUs, estimated), 4,730 * 30 = 141,900 seconds(~40 hours)

---

## Acknowledgments

This project was inspired by the implementation of **GPT-2 124M**, from which the initial pre-training script was adapted.  

### 🔹 **Original GPT-2 Code Reference**  
- **Author**: ***Andrej karpathy***  
- **Repository**: https://github.com/karpathy/build-nanogpt
- **Files Directly Taken from This Repository**:
  - **hellaswag.py**: Used for evaluating the model on the HellaSwag benchmark.
  - **fineweb.py**: Used for installing and preparing the FineWeb-Edu dataset.

### 🔹 **Modifications Made**:
- Adapted the architecture from **GPT-2 124M** to **Llama2 7B**.
- Replaced **LayerNorm** with **RMSNorm**, as used in the Llama architecture.
- Integrated **KV caching** for more efficient text generation during pre-training.
- Added **Rotary Positional Embeddings (RoPE)** following the Llama architecture.
- Modified the **FeedForward** layer to align with the **Llama architecture**, replacing GeLU with **SwiGLU**.

Special thanks to **Andrej karpathy** for his open-source work, which provided a strong foundation for this project.

---

## Contact  

For any questions or inquiries, feel free to reach out:  

- **Name**: *Khaled Alharthi*
- **GitHub**: [GitHub Profile](https://github.com/khalharthi)  
- **Email**: khalharthi1991@outlook.com
- **LinkedIn**: [LinkedIn](https://www.linkedin.com/in/khaled-alharthi-5b7532220/)  

Feel free to open an issue in this repository if you encounter any problems or have feature requests!

