# Llama2 7B Pre-training Scripts

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains the scripts for pre-training the Llama2 7B language model from scratch using the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset. The project focuses on efficient generation using KV (Key-Value) caching and periodic evaluation using the Hellaswag benchmark.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Evaluation](#evaluation)
- [Folder Structure](#folder-structure)
- [Tokens and Batches per GPU](#tokens-and-batches-per-gpu)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features
- **Llama2 7B Pre-training Scripts**: Contains the code to pre-train the Llama2 7B model from scratch using the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset.
- **KV Cache for Efficient Generation**: Implements KV caching during generation to reduce memory usage and improve inference speed.
- **Periodic Generation During Training**: After every 250 steps, the script generates text using the prompt: `"Hello, I'm a large language model, "` and produces up to 40 tokens to monitor model progress.
- **Hellaswag Evaluation**: Evaluates the model using the Hellaswag benchmark every 250 steps and at the final step to measure commonsense reasoning capabilities.

---

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/khalharthi/llama2-7b-pretraining.git
   cd llama2-7b-pretraining
