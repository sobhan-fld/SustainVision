SustainVision: Green AI Training Platform
========================================

SustainVision is an innovative, user-centric Machine Vision (MV) training platform
dedicated to advancing the practice of Green AI. It empowers developers,
researchers, and students to quantify and minimize the environmental impact of
their deep learning projects.

The repository currently focuses on the interactive configuration layer that
orchestrates model selection, dataset setup, hardware targeting, and
hyperparameter tuning. The configuration is persisted so future training runs can
share the same baseline—an essential step toward reproducible, low-impact
experiments.

Key Capabilities
----------------

- Command-line TUI built with `questionary` for editing and saving training configs
- YAML-backed `ConfigManager` with sensible defaults and merge-friendly reloads
- Automatic discovery of available devices (CPU and CUDA indices when PyTorch is installed)
- Built-in reference training loop (PyTorch) wrapped with CodeCarbon emissions tracking
- CSV reporting pipeline capturing per-epoch metrics plus sustainability summary
- Real training loop leveraging torchvision backbones (ResNet, MobileNet,
  EfficientNet, ViT) with configurable datasets and CodeCarbon logging
- Advanced training controls: optimizer/loss selection (Cross-Entropy, SimCLR,
  SupCon, etc.), weight decay, schedulers, AMP toggle, gradient clipping, and
  reproducibility seed management
- Dataset download helper that stores assets under `databases/` (ignored by git)

Planned Enhancements
--------------------

- Provide starter training loops for common MV workloads/datasets
- Surface sustainability metrics directly in the TUI and in richer dashboards
- Add dataset management helpers (downloads, splits, versioning)

Getting Started
---------------

1. **Clone the repo** (if you haven’t already):
   ```bash
   git clone https://github.com/your-org/SustainVision.git
   cd SustainVision
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Required: `questionary`, `PyYAML`, `codecarbon`, `numpy`
   - Optional (but recommended): `torch` for training + CUDA device detection
   - Optional: `torchvision` for dataset downloads (CIFAR10, MNIST, ...)
   - Optional (Windows): `colorama` for vibrant CLI colors
   - Optional (advanced optimizers): `lion-pytorch` when using the Lion optimizer

Using the TUI
-------------

Launch the interactive configuration flow:

```bash
python main.py
```

The main menu offers:

- **Start training** – runs the reference PyTorch loop, logging metrics & emissions
- **Configure settings** – launches the detailed config editor
- **Download databases** – choose a dataset and save it under `databases/`
- **Show current config** – prints the resolved config (from disk + defaults)

Configuration changes persist automatically to:

- Linux: `~/.config/sustainvision/config.yaml` (or `$XDG_CONFIG_HOME/sustainvision/config.yaml`)

Training Workflow & Output
--------------------------

- The sample training loop creates a synthetic classification dataset and trains a
- The training loop now consumes real datasets (CIFAR10, MNIST, or custom
  `ImageFolder` layouts) and trains torchvision backbones with projection heads.
- Contrastive objectives (SimCLR / SupCon) use strengthened augmentations and
  normalized projections, while Cross-Entropy-based losses follow standard
  classification.
- CodeCarbon wraps the full run and reports total energy usage and CO₂e emissions.
- Metrics and sustainability data are written to a CSV file in the project root.
  - The file name comes from `report_filename` in the config (default
    `training_report.csv`).
  - If a file with the same name already exists, an index is appended (e.g.,
    `training_report_1.csv`).
  - Columns include: epoch, train/val loss & accuracy, emissions, energy, run
    duration, and config metadata (model/database/device).

Fine-Tuning Workflow
--------------------

For self-supervised learning (SimCLR/SupCon), use a two-stage approach:

1. **Pre-training**: Train with contrastive loss to learn representations
   - Set `loss_function: simclr` or `supcon`
   - Enable model saving to get a checkpoint
   - Early stopping automatically uses `train_loss` (not `val_accuracy`)

2. **Fine-tuning**: Load the checkpoint and train with classification loss
   - Set `checkpoint_path` to your saved checkpoint (e.g., `resnet18_simclr_checkpoints/resnet18_simclr_model.pt`)
   - Change `loss_function` to `cross_entropy`
   - Optionally set `freeze_backbone: True` for linear evaluation (faster, evaluates representation quality)
   - Or leave `freeze_backbone: False` for full fine-tuning (usually better accuracy)

Example: After training ResNet18 with SimCLR, fine-tune it:
- `checkpoint_path: resnet18_simclr_checkpoints/resnet18_simclr_model.pt`
- `loss_function: cross_entropy`
- `freeze_backbone: False` (or `True` for linear eval)

Programmatic Access
-------------------

Import the `ConfigManager` whenever you need programmatic read/write access:

```python
from sustainvision.config import ConfigManager

cm = ConfigManager()
cfg = cm.load()
print(cfg.device)
```

Configuration Reference
-----------------------

- `model`: String identifier (e.g., `resnet18`, `vit-base`, custom)
- `database`: Dataset alias or filesystem path
- `device`: `cpu` or `cuda:{index}`
- `report_filename`: Desired CSV filename for training metrics & emissions summary
- `seed`: Random seed applied to Python, NumPy, and PyTorch
- `optimizer`: Optimizer name (`adam`, `sgd`, `adamw`, `rmsprop`, `lion`, ...)
- `loss_function`: Loss function (`cross_entropy`, `mse`, `bce`, `simclr`, `supcon`, ...)
- `weight_decay`: L2 regularization strength
- `scheduler`: Learning rate scheduler descriptor (`type` + `params`)
- `gradient_clip_norm`: Optional gradient clipping threshold
- `mixed_precision`: Toggle automatic mixed precision (AMP)
- `save_model`: `True/False` to store the trained weights after each run
- `save_model_path`: Directory where checkpoints are written (default `artifacts/`)
- `checkpoint_path`: Optional path to a checkpoint file for fine-tuning (leave empty for training from scratch)
- `freeze_backbone`: If fine-tuning, freeze the encoder/backbone and only train the classifier head
- `early_stopping`: Configuration dict with `enabled`, `patience`, `metric`, and `mode`
  - Note: For contrastive losses (SimCLR/SupCon), `val_accuracy` is automatically replaced with `train_loss`
- `hyperparameters`
  - `batch_size`
  - `lr`
  - `epochs`
  - `momentum`
  - `temperature`
  - `num_workers`
  - `val_split`
  - `image_size`
  - `projection_dim`

Available Options Cheat Sheet
-----------------------------

**Models**
- `resnet18`, `resnet34`, `resnet50`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `vit_b_16`
- Any custom model string (falls back to an MLP classifier if not recognized)

**Loss Functions**
- `cross_entropy` (single-label classification)
- `binary_cross_entropy` (auto one-hot for multi-class)
- `mse`, `l1`, `smooth_l1`
- `simclr`, `supcon` (contrastive)

**Optimizers**
- `adam`, `adamw`, `sgd`, `rmsprop`, `lion`
- Custom strings fall back to Adam

**Schedulers**
- `none` (default)
- `step_lr` (`step_size`, `gamma`)
- `cosine_annealing` (`t_max`, `eta_min`)
- `exponential` (`gamma`)

**Devices**
- `cpu`, `cuda`, or `cuda:{index}`. On Colab/GPUs, set `device="cuda"` so the trainer stays on GPU.

**Dataset Helpers**
- `cifar10`, `mnist`, `synthetic`
- Any ImageFolder layout (`train/`, `val/`, `test/` directories)

**Early Stopping**
- Toggle on/off and configure with:
  - `patience` (epochs without improvement)
  - `metric`: `val_loss`, `val_accuracy`, `train_loss`, `train_accuracy`
  - `mode`: `min` or `max`

**TUI Prompts**
- Every prompt mirrors the config keys above; defaults are safe for CPU smoke tests (batch 32, 1 epoch, mobilenet, etc.).

Dataset Storage
---------------

- All downloaded or custom datasets should live under the project `databases/`
  directory. The folder is git-ignored by default (a `.gitkeep` file keeps the
  directory present in version control).
- After downloading via the menu, the configuration automatically updates to
  point to the freshly prepared dataset path.

Dependencies Overview
---------------------

- `questionary` – interactive CLI prompts
- `PyYAML` – configuration persistence
- `codecarbon` – energy and emissions estimation
- `numpy` – numeric utilities (seeding, parsing)
- `torch` (optional) – reference training implementation + CUDA detection
- `torchvision` (optional) – dataset downloads for common CV benchmarks
- `lion-pytorch` (optional) – Lion optimizer support
- `colorama` (optional) – improved Windows terminal colors

Contributing & Feedback
-----------------------

- Issues and feature requests are welcome—especially around sustainability
  metrics, dataset integrations, and carbon accounting workflows.
- Contributions should follow standard Python best practices (linting, typing
  where feasible, and clear docstrings for new components).

License
-------

Please add a license file to clarify usage (MIT, Apache 2.0, etc.).

