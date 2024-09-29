# TL-CL: Task And Language Incremental Continual Learning

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Training Methods](#training-methods)
5. [Evaluation](#evaluation)
6. [Configuration](#configuration)

## Installation

Our implementation is based on PyTorch and HuggingFace (transformers + datasets).

1. Make sure you have Anaconda installed. If not, follow this [miniconda installation](https://docs.conda.io/en/latest/miniconda.html) guide.

2. To run code on GPU, ensure you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. Our implementation uses CUDA 11.0.

3. Re-create our conda environment from the `requirements.yaml` file:

```bash
conda env create -f requirements.yaml
```

This process can take about 15-20 minutes.

4. Activate the environment:

```bash
conda activate TLCL
```

5. Download the required data and place it in the home folder. Each training script contains a path for the root data directory.

## Project Structure

The project is organized as follows:

- `main.py`: The main entry point for training and evaluation.
- `multitask_training.py`: Contains code for multitask training.
- `inference.py`: Handles model inference and evaluation.
- `dataset.py`: Defines the dataset classes used in the project.
- `metrics.py`: Contains functions for computing evaluation metrics.
- `utils.py`: Utility functions for various tasks.
- `TLSA/`: Directory containing Task and Language Specific Adapters implementation.
- `baselines/`: Directory containing implementations of baseline methods.
- `scripts/`: Contains training scripts for various methods.

## Usage

To run any training method:

```bash
cd scripts
bash <method_name>.sh <GPU ID> <PORT NO> <SEQ>
```

For Task and Language Specific Adapters (TLSA):

```bash
cd ./TLSA/script
source <script_name>.sh <GPU ID> <PORT NO> <SEQ>
```

Note: We use three GPUs at once. The port number is needed to avoid conflicts when running multiple experiments in parallel. The `SEQ` parameter denotes the task sequence.

## Training Methods

The project implements several continual learning methods:

1. Sequential Fine-tuning
2. EWC (Elastic Weight Consolidation)
3. ER (Experience Replay)
4. AT<sub>ER</sub> (Adapter Finetuning with Experience Replay)
5. AT (Parameter Isolation Based Adapter Finetuning)
6. MAD-X (Modular Adapters for Cross-lingual NLP)
7. MTMLFT (Multiligual Multitask Finetuning)
8. **TLSA (Task and Language Specific Adapters)** (Proposed Approach)

Each method is implemented in the `baselines/` directory or the `TLSA/` directory.

## Evaluation

Evaluation is performed using various metrics depending on the task:

- Classification: Exact Match (EM)
- Natural Language Inference: Exact Match (EM)
- Question Answering: F1 Score (F1)
- Summarization: ROUGE scores

The `eval` function in `inference.py` handles the evaluation process for different tasks and languages.

## Configuration

The main configuration options are set through command-line arguments in `main.py`. Key parameters include:

- Model name or checkpoint path
- Training method
- Task sequence
- Number of epochs
- Batch size
- Learning rate
- Output and data directories

For TLSA-specific configurations, refer to the `TLSA/main.py` file.

You can modify hyperparameters in the training scripts or in `main.py` and `multitask_training.py` for multitask training.

For more detailed information about specific components, please refer to the individual Python files in the project.