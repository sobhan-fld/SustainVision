SustainVision: Green AI Training Platform
========================================

Maintained by **Sobhan Fooladi Mahani (CTTC)**.

SustainVision is an innovative, user-centric Machine Vision (MV) training platform
dedicated to advancing the practice of Green AI. It empowers developers,
researchers, and students to quantify and minimize the environmental impact of
their deep learning projects.

The repository provides an interactive configuration layer that orchestrates
model selection, dataset setup, hardware targeting, and hyperparameter tuning.
The configuration is persisted so future training runs can share the same
baseline—an essential step toward reproducible, low-impact experiments. The
codebase is modular, with training logic organized into focused modules for
losses, optimizers, models, reporting, and schedule management.

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
- **Contrastive learning alternating schedule**: Automate pretrain/finetune cycles
  for SimCLR/SupCon with configurable epochs per phase, learning rates, and
  backbone freezing options
- Dataset download helper that stores assets under `databases/` (ignored by git)
- Modular codebase: Training logic split into focused modules (utils, losses,
  optimizers, models, reporting, schedule) for better maintainability

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
- Columns now include: epoch, phase (e.g., `pretrain_cycle_1`, `finetune_cycle_1`),
  `loss_name`, train/val loss & accuracy, learning rate, emissions, energy,
  run duration, config metadata, plus dedicated views of the current objective:
  - `contrastive_*` columns are filled only for SimCLR/SupCon epochs
  - `classifier_*` columns are filled for supervised/finetune epochs
  - This keeps contrastive metrics visible without confusing them with classifier accuracy.
  - For alternating schedules, all phases are logged in a single CSV with
    continuous epoch numbering and phase labels.

Fine-Tuning Workflow
--------------------

For self-supervised learning (SimCLR/SupCon), you have two options:

### Option 1: Manual Two-Stage Approach

1. **Pre-training**: Train with contrastive loss to learn representations
   - Set `loss_function: simclr` or `supcon`
   - Enable model saving to get a checkpoint
   - Early stopping automatically uses `train_loss` (not `val_accuracy`)

2. **Fine-tuning**: Load the checkpoint and train with classification loss
   - Set `checkpoint_path` to your saved checkpoint (e.g., `artifacts/model_cycle1.pt`)
   - Change `loss_function` to `cross_entropy`
   - Optionally set `freeze_backbone: True` for linear evaluation (faster, evaluates representation quality)
   - Or leave `freeze_backbone: False` for full fine-tuning (usually better accuracy)

### Option 2: Automated Alternating Schedule (Recommended)

Enable the contrastive learning alternating schedule to automatically run
pretrain/finetune cycles:

1. **Enable the schedule** in the TUI:
   - Set `simclr_schedule.enabled: true`
   - Configure cycles (e.g., 8 cycles of 50 pretrain + 20 finetune epochs)
   - Set pretrain loss (`simclr` or `supcon`) and finetune loss (`cross_entropy`)
   - Optionally set a different learning rate for finetune phase
   - Choose whether to freeze backbone during finetune (linear evaluation)

2. **The schedule automatically**:
   - Runs pretrain phase (contrastive learning)
   - Runs finetune phase (supervised learning)
   - Saves checkpoints after each finetune phase (`_cycle1.pt`, `_cycle2.pt`, ...)
   - Continues epoch numbering across all phases
   - Tracks emissions for the entire run
- When `freeze_backbone: true`, the classifier head is re-initialized before each
  finetune phase, mirroring the “train backbone → train linear head” procedure
  recommended in SimCLR/SupCon workflows.

**Example configuration**:
```yaml
simclr_schedule:
  enabled: true
  cycles: 8
  pretrain_epochs: 50
  finetune_epochs: 20
  pretrain_loss: supcon  # or simclr
  finetune_loss: cross_entropy
  finetune_lr: 0.01  # Optional: different LR for finetune
  freeze_backbone: true  # Linear evaluation
  optimizer_reset: true  # Reset optimizer between phases

hyperparameters:
  batch_size: 256
  use_m_per_class_sampler: true  # Recommended for SupCon
  m_per_class: 8  # 32 classes * 8 samples = 256 batch size
```

This gives you 8 cycles × (50 + 20) = 560 total epochs with automatic phase switching.

### Option 3: Multi-Head Evaluation (Train Once, Evaluate Multiple Tasks)

After training a feature space (e.g., SupCon with MobileNet Small on CIFAR-10), you can
evaluate the same pretrained backbone with different task-specific heads without
retraining the backbone:

1. **Train your feature space** using Option 1 or Option 2 above
2. **Enable evaluation mode** in your config:
   ```yaml
   evaluation:
     enabled: true
     checkpoint_path: outputs/your_model_cycle1.pt
     head_type: classification  # or "detection"
   ```
3. **Run evaluation** from the main menu: "Evaluate pretrained model"

The evaluation system:
- Loads your pretrained backbone (frozen)
- Attaches a task-specific head (classification or object detection)
- Trains only the head while keeping the backbone frozen
- Reports evaluation metrics

**Example workflow:**
```yaml
# Step 1: Train feature space with SupCon
model: mobilenet_v3_small
database: databases/cifar10
loss_function: supcon
simclr_schedule:
  enabled: true
  cycles: 10
  pretrain_epochs: 100
  pretrain_loss: supcon

# Step 2: Evaluate with classification head
evaluation:
  enabled: true
  checkpoint_path: outputs/mobilenet_small_supcon_cifar10_cycle1.pt
  head_type: classification

# Step 3: Evaluate with detection head (same backbone!)
evaluation:
  enabled: true
  checkpoint_path: outputs/mobilenet_small_supcon_cifar10_cycle1.pt
  head_type: detection
  num_anchors: 9
  hidden_dim: 256
```

This allows you to train the feature space once and evaluate it on multiple downstream
tasks, demonstrating the transferability of learned representations.

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
- `quantization`: Optional quantized export configuration (enable/disable, approach, dtype, backend, artifact format)
- `save_model`: `True/False` to store the trained weights after each run
- `save_model_path`: Directory where checkpoints are written (default `artifacts/`)
- `checkpoint_path`: Optional path to a checkpoint file for fine-tuning (leave empty for training from scratch)
- `freeze_backbone`: If fine-tuning, freeze the encoder/backbone and only train the classifier head
- `early_stopping`: Configuration dict with `enabled`, `patience`, `metric`, and `mode`
  - Note: For contrastive losses (SimCLR/SupCon), `val_accuracy` is automatically replaced with `train_loss`
- `simclr_schedule`: Contrastive learning alternating schedule configuration
  - `enabled`: Enable/disable the schedule (default: `false`)
  - `cycles`: Number of pretrain+finetune cycles (default: `8`)
  - `pretrain_epochs`: Epochs per pretrain phase (default: `50`)
  - `finetune_epochs`: Epochs per finetune phase (default: `20`)
  - `pretrain_loss`: Loss for pretrain phase (`simclr` or `supcon`, default: `simclr`)
  - `finetune_loss`: Loss for finetune phase (default: `cross_entropy`)
  - `finetune_lr`: Learning rate for finetune (default: `None`, uses same as pretrain)
  - `freeze_backbone`: Freeze backbone during finetune for linear evaluation (default: `false`)
  - `optimizer_reset`: Reset optimizer between phases (default: `true`)
- `hyperparameters`
  - `batch_size`
  - `lr`
  - `epochs`
  - `momentum`
  - `lars_eta`, `lars_eps`, `lars_exclude_bias_n_norm` (only used when `optimizer="lars"`)
  - `temperature`
  - `num_workers`
  - `val_split`
  - `image_size`
  - `projection_dim`
  - `projection_hidden_dim` (optional, for multi-layer projection head)
  - `projection_use_bn` (optional, enable batch normalization in projection head)
  - `use_m_per_class_sampler` (optional, for SupCon: ensures balanced batches with positive pairs)
  - `m_per_class` (optional, number of samples per class per batch when using MPerClassSampler)

Available Options Cheat Sheet
-----------------------------

**Models**
- `resnet18`, `resnet34`, `resnet50`
- `mobilenet_v3_small`, `mobilenet_v3_large`
- `efficientnet_b0`
- `vit_b_16`
- Any custom model string (falls back to an MLP classifier if not recognized)

**Loss Functions**
- `cross_entropy` (single-label classification)
- `binary_cross_entropy` (auto one-hot for multi-class)
- `mse`, `l1`, `smooth_l1`
- `simclr`, `supcon` (contrastive)

**Optimizers**
- `adam`, `adamw`, `sgd`, `rmsprop`, `lion`, `lars`
- Custom strings fall back to Adam

**Schedulers**
- `none` (default)
- `step_lr` (`step_size`, `gamma`)
- `cosine_annealing` (`t_max`, `eta_min`)
- `warmup_cosine` (`warmup_epochs`, `warmup_start_factor`, `t_max`, `eta_min`)
- `exponential` (`gamma`)

**Devices**
- `cpu`, `cuda`, or `cuda:{index}`. On Colab/GPUs, set `device="cuda"` so the trainer stays on GPU.

**Dataset Helpers**
- `cifar10`, `cifar100`, `mnist`, `synthetic`
- Any ImageFolder layout (`train/`, `val/`, `test/` directories)

**Early Stopping**
- Toggle on/off and configure with:
  - `patience` (epochs without improvement)
  - `metric`: `val_loss`, `val_accuracy`, `train_loss`, `train_accuracy`
  - `mode`: `min` or `max`

**Contrastive Learning Schedule**
- Enable alternating pretrain/finetune cycles
- Configure number of cycles, epochs per phase, loss functions
- Set separate learning rate for finetune phase
- Choose to freeze backbone during finetune (linear evaluation)
- Checkpoints saved after each finetune phase with cycle suffix

**MPerClassSampler for SupCon**
- When using SupCon loss, enable `use_m_per_class_sampler: true` to ensure each batch contains balanced samples from multiple classes
- This guarantees positive pairs exist in every batch, which is crucial for effective SupCon learning
- Set `m_per_class` to the number of samples per class per batch (e.g., `m_per_class: 8` with `batch_size: 256` gives 32 classes per batch)
- Automatically disabled for non-contrastive losses

**TUI Prompts**
- Every prompt mirrors the config keys above; defaults are safe for CPU smoke tests (batch 32, 1 epoch, mobilenet, etc.).
- Schedule configuration prompts appear when enabling the contrastive learning schedule.

Dataset Storage
---------------

- All downloaded or custom datasets should live under the project `databases/`
  directory. The folder is git-ignored by default (a `.gitkeep` file keeps the
  directory present in version control).
- After downloading via the menu, the configuration automatically updates to
  point to the freshly prepared dataset path.

Code Structure
--------------

The codebase is organized into focused modules for maintainability:

- **`config.py`**: Configuration management with YAML persistence
- **`tui.py`**: Interactive text user interface for configuration
- **`data.py`**: Dataset loading and download utilities
- **`training.py`**: Core training loop and orchestration
- **`schedule.py`**: Contrastive learning alternating schedule implementation
- **`utils.py`**: Utility functions (device resolution, seeding, paths)
- **`losses.py`**: Loss function implementations (SimCLR, SupCon, standard losses)
- **`samplers.py`**: Custom data samplers (MPerClassSampler for balanced contrastive learning batches)
- **`optimizers.py`**: Optimizer and scheduler builders
- **`models.py`**: Model building utilities (torchvision backbones, projection heads)
- **`reporting.py`**: CSV reporting and metrics logging

This modular structure makes it easy to:
- Understand what each component does
- Test individual components
- Extend functionality (e.g., add new loss functions or optimizers)
- Maintain and debug the codebase

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

We welcome issues, feature requests, and success stories. Tell us how you are using SustainVision, where the workflows feel clunky, or what sustainability metrics you would love to track next. If you plan to open a pull request, please follow common Python conventions (linting, type hints where practical, and clear docstrings that explain why a change exists).


Example Commands for Running from Config
-----------------------------------------

**CIFAR100 with ResNet18 and SupCon:**
```bash
cd /home/sfooladi/github/SustainVision
PYTHONPATH=$(pwd) ~/.virtualenvs/SustainVision/bin/python scripts/run_with_config.py \
  --config configs/cifar100_resnet18_supcon_full_v3.yaml \
  --project-root outputs \
  | tee logs/supcon_resnet18_cifar100_$(date +%Y%m%d_%H%M%S).log
```

**CIFAR100 with MobileNet V3 Large and SupCon:**
```bash
cd /home/sfooladi/github/SustainVision
PYTHONPATH=$(pwd) ~/.virtualenvs/SustainVision/bin/python scripts/run_with_config.py \
  --config configs/cifar100_mobilenet_large_supcon_full_v3.yaml \
  --project-root outputs \
  | tee logs/supcon_mbv3large_cifar100_$(date +%Y%m%d_%H%M%S).log
```

**CIFAR100 with EfficientNet-B0 and SupCon:**
```bash
cd /home/sfooladi/github/SustainVision
PYTHONPATH=$(pwd) ~/.virtualenvs/SustainVision/bin/python scripts/run_with_config.py \
  --config configs/efficientnet_b0_supcon_full_v3.yaml \
  --project-root outputs \
  | tee logs/supcon_efficientnet_cifar100_$(date +%Y%m%d_%H%M%S).log
```

License
-------

Released under the MIT License. See the `LICENSE` file for details.

