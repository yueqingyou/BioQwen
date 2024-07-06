# README

## Overview

This repository contains the code for training a QLoRA-based model (`BioQwen`) with two stages. The model is trained using data from various sources, filtered, tokenized, and prepared for causal language modeling. The training is performed using the `transformers` library from Hugging Face, and the model utilizes the LoRA (Low-Rank Adaptation) and QLoRA techniques for efficient training.

## Prerequisites

- Python 3.8 or higher
- PyTorch
- bitsandbytes
- gradio
- datasets
- transformers
- peft
- trl

Install the required packages using:
```bash
pip install torch bitsandbytes gradio datasets transformers peft trl
```
or
```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── data/
│   ├── COIG-CQIA/
│   ├── Taiyi_Instruction_Data_001/
│   ├── MedChatZH/
│   ├── self_cognition.json
├── model/
│   ├── BioQwen-stage1-merged/
│   ├── BioQwen-stage1/
│   ├── BioQwen-stage2/
├── script_stage1.py
├── script_stage2.py
└── README.md
```

## Data

The data is stored in the `data` directory and includes various datasets:
- `COIG-CQIA`
- `Taiyi_Instruction_Data_001`
- `MedChatZH`
- `self_cognition.json`

## Training

The training is performed in two stages, as described in `script_stage1.py` and `script_stage2.py`.

### Stage 1 Training

1. **Load and Filter Data:**
    - Load datasets.
    - Filter and preprocess the data.
    - Tokenize the data with language checks.

2. **Model Setup:**
    - Load the pre-trained model and tokenizer.
    - Configure BitsAndBytes for efficient training.
    - Prepare the model for QLoRA/LoRA training.

3. **Training Configuration:**
    - Define training arguments.
    - Create a Trainer instance and start training.

### Stage 2 Training

1. **Load and Filter Data:**
    - Load additional datasets.
    - Concatenate datasets for stage 2 training.
    - Tokenize the data with language checks.

2. **Model Setup:**
    - Load the model from stage 1.
    - Configure BitsAndBytes for efficient training.
    - Prepare the model for QLoRA/LoRA training.

3. **Training Configuration:**
    - Define training arguments.
    - Create a Trainer instance and start training.

## Running the Training

### Single GPU

To run the training on a single GPU, use the `python` command:

```bash
python script_stage1.py
python script_stage2.py
```

### Multiple GPUs

To run the training on multiple GPUs, use the `torchrun` command:

```bash
torchrun --nproc_per_node=NUM_GPUS script_stage1.py
torchrun --nproc_per_node=NUM_GPUS script_stage2.py
```

Replace `NUM_GPUS` with the number of GPUs available on your machine.

## Scripts

### script_stage1.py

This script handles the first stage of training. It includes loading and filtering data, preparing the model, and training.

### script_stage2.py

This script handles the second stage of training. It includes loading and filtering additional data, preparing the model, and training.

## Notes

- Ensure that the `model_path` in both scripts points to the correct model directory.
- The training process may require a significant amount of computational resources, especially for large datasets and models.
- Adjust the training arguments such as batch size, learning rate, and number of epochs based on your specific requirements and available resources.

## Contact

For any questions or further information, please submit an issue on this repository.