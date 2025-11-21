# Phase 01: Environment Setup & Data Acquisition

## Objective
Set up the development environment locally and prepare for Google Colab execution with GPU support and necessary dependencies.

## Duration
0.5 hours

## Tasks

### 1.1 Project Structure Creation
```bash
ADetective/
├── data/
│   └── .gitignore  # Ignore large data files
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   ├── transformer.py
│   │   └── scgpt_wrapper.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── metrics.py
├── configs/
│   ├── mlp_config.yaml
│   ├── transformer_config.yaml
│   └── scgpt_config.yaml
├── notebooks/
│   └── colab_runner.ipynb  # Example Colab notebook
├── scripts/
│   ├── train_mlp.py
│   ├── train_transformer.py
│   └── train_scgpt.py
├── results/
│   └── .gitignore  # Ignore result files
├── requirements.txt
├── requirements_colab.txt  # Colab-specific if needed
├── setup.py  # For package installation
├── README.md
└── .gitignore
```

### 1.2 Requirements File Creation
Create `requirements.txt`:
```txt
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Single-cell analysis
scanpy>=1.9.0
anndata>=0.8.0

# Deep learning
torch>=2.0.0
accelerate>=0.24.0

# Configuration
pyyaml>=6.0
tqdm>=4.62.0

# Evaluation
scikit-learn>=1.0.0
```

Create `requirements_scgpt.txt` (separate due to potential conflicts):
```txt
# Foundation model specific
scgpt>=0.1.0  # If available via pip
# Or instructions to install from GitHub
flash-attn>=2.0.0  # For Flash Attention support
transformers>=4.30.0
```

### 1.3 Setup Script
Create `setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="adetective",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt')
        if not line.startswith('#')
    ],
    python_requires=">=3.8",
)
```

### 1.4 Configuration System
Create `src/utils/config.py`:
```python
import os
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path=None):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.configs_dir = self.project_root / "configs"

        # Check if running in Colab
        self.in_colab = 'COLAB_GPU' in os.environ

        if self.in_colab:
            # Adjust paths for Colab
            self.data_dir = Path("/content/drive/MyDrive/ADetective/data")
            self.results_dir = Path("/content/drive/MyDrive/ADetective/results")

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(self, key, value)
```

### 1.5 Git Configuration
Create `.gitignore`:
```
# Data files
*.h5ad
*.h5
data/raw/
data/processed/

# Results
results/
*.pkl
*.pth
*.pt

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# Environment
venv/
ENV/
env/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

### 1.6 Colab Setup Script
Create `scripts/setup_colab.sh`:
```bash
#!/bin/bash

# Mount Google Drive
echo "Mounting Google Drive..."

# Clone repository if not exists
if [ ! -d "/content/ADetective" ]; then
    echo "Cloning repository..."
    git clone https://github.com/[username]/ADetective.git /content/ADetective
fi

cd /content/ADetective

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo "Setup complete!"
```

## Validation Checklist
- [ ] Project structure created with all directories
- [ ] Requirements files created with correct dependencies
- [ ] Configuration system supports both local and Colab environments
- [ ] Git properly configured with .gitignore
- [ ] Setup scripts ready for Colab execution

## Next Steps
After completing Phase 01:
1. Push initial structure to GitHub
2. Test cloning and setup in Google Colab
3. Verify GPU access and dependency installation
4. Proceed to Phase 02 for data acquisition and exploration

## Notes
- Data files should be stored in Google Drive for Colab access
- Consider using wandb or tensorboard for experiment tracking
- Ensure all paths are configurable for different environments