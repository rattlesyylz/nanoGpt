# NanoGPT Implementation

A lightweight implementation of GPT architecture for language modeling.

## Project Structure

- `model.py`: GPT model implementation
- `train.py`: Training infrastructure
- `inference.py`: Text generation utilities
- `sanity_check.py`: Validation tests for model functionality

## Features

- Character-level language modeling
- Transformer-based architecture
- Configurable model parameters (layers, heads, embedding dimensions)
- Support for masked loss computation
- Text generation with temperature control

## Usage

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py test [model_path]
```

### Running Sanity Checks

```bash
python sanity_check.py
``` 