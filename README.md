# ADetective: Oligodendrocyte AD Pathology Classifier

Binary classification of Alzheimer's Disease (AD) pathology in oligodendrocytes using single-cell RNA-seq data.

## Overview

This project implements multiple machine learning approaches to classify AD pathology status (High vs Not AD) in oligodendrocytes from the SEAAD dataset:

1. **MLP Baseline**: Simple fully-connected neural network baseline
2. **Custom Transformer**: Transformer-based model with gene embeddings
3. **scGPT Fine-tuning**: Foundation model fine-tuning approach

## Dataset

- **Source**: SEAAD (Single-nucleus Enrichment and Sequencing of Alzheimer Disease snRNAseq)
- **File**: `SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad` (35GB)
- **Format**: AnnData H5AD

## Installation

### Local Setup

```bash
# Clone repository
git clone <repo-url>
cd ADetective

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### scGPT Setup (Optional)

```bash
pip install -r requirements_scgpt.txt
cd src/models
pip install -e .
```

### Google Colab

```python
!git clone <repo-url>
%cd ADetective
!pip install -r requirements.txt
```

## Quick Start

### Data Preparation

```bash
python scripts/prepare_data.py
```

### Train Models

```bash
# MLP Baseline
python scripts/training/train_mlp.py

# Transformer
python scripts/training/train_transformer.py

# scGPT Fine-tuning
python scripts/training/train_scgpt.py
```

### Evaluate Models

```bash
python scripts/evaluate.py
```

## Project Structure

```
ADetective/
├── src/
│   ├── models/           # Model implementations
│   ├── utils/            # Utilities (config, logging, etc.)
│   ├── data/             # Data loading and preprocessing
│   └── eval/             # Evaluation metrics and functions
├── scripts/
│   ├── training/         # Training scripts
│   ├── inference/        # Inference scripts
│   └── prepare_data.py   # Data preparation
├── configs/              # Configuration YAML files
├── results/              # Output directory for results
├── tests/                # Unit tests
├── requirements.txt      # Main dependencies
├── requirements_scgpt.txt # scGPT-specific dependencies
└── setup.py              # Package setup
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy | >85% |
| F1 Score | >0.80 |
| ROC-AUC | >0.85 |

## Contributing

Please ensure code quality with:

```bash
black src/
isort src/
flake8 src/
pytest tests/
```

## Citation

If you use this project in your research, please cite:

```bibtex
@software{adetective2024,
  title={ADetective: Oligodendrocyte AD Pathology Classifier},
  author={Research Team},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details

## Resources

- [SEAAD Dataset](https://www.synapse.org/)
- [scGPT Documentation](https://scgpt-guide.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

*This project is part of a multi-phase ML/AI research initiative.*
