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

Detection Variants (Evaluation Mode)
------------------------------------

Set `evaluation.enabled: true` and choose one of these `evaluation.head_type` values:

- `detection`: SustainVision slot-based detection head (lightweight representation evaluation)
- `torchvision_frcnn`: torchvision Faster R-CNN with ResNet18 + FPN
- `torchvision_rcnn`: Faster R-CNN using a single C4 feature map (R-CNN-like)
- `rcnn_classic`: experimental classic R-CNN style pipeline (grid proposals + ROI crops)

Example snippets:

```yaml
evaluation:
  enabled: true
  checkpoint_path: outputs/your_checkpoint.pt
  head_type: detection
  num_anchors: 9
  hidden_dim: 256
```

```yaml
evaluation:
  enabled: true
  checkpoint_path: outputs/your_checkpoint.pt
  head_type: torchvision_frcnn
  nms_threshold: 0.5
```

```yaml
evaluation:
  enabled: true
  checkpoint_path: outputs/your_checkpoint.pt
  head_type: torchvision_rcnn
  nms_threshold: 0.5
```

```yaml
evaluation:
  enabled: true
  checkpoint_path: outputs/your_checkpoint.pt
  head_type: rcnn_classic
  freeze_backbone: true
  hidden_dim: 256
  rcnn_classic_roi_size: 96
  rcnn_classic_max_grid_proposals: 96
```

Expected detection dataset layouts:

- COCO: `databases/coco/images/train2017`, `databases/coco/images/val2017`, `databases/coco/annotations/*.json`
- Pascal VOC: torchvision layout `databases/voc/VOCdevkit/VOC2012/...` or XML/Kaggle-style `Annotations/`, `JPEGImages/`, `ImageSets/`

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

Developer Guide (Architecture & File Map)
-----------------------------------------

This section is intended for contributors who want to understand how the codebase
is organized and where to implement changes safely.

### Core design principles

- **Config-driven orchestration**: Most workflows are controlled by `TrainingConfig`
  values loaded through `ConfigManager`, so new features should prefer config flags
  over hard-coded branches.
- **Separation of concerns**: Data loading, model building, optimization, metrics,
  and orchestration are split into dedicated modules.
- **Optional dependency resilience**: Many modules support partial functionality when
  `torch`, `torchvision`, `codecarbon`, or other optional packages are unavailable.
- **Evaluation as a dispatcher**: `sustainvision/evaluation.py` routes to task-specific
  evaluation loops while preserving a stable public entrypoint.
- **Artifact-first experimentation**: Training/evaluation paths are expected to save
  enough metadata (config, CSV metrics, checkpoints, emissions data) to reproduce runs.

### Repository map (top level)

- `main.py`
  - Interactive entrypoint (Questionary menu).
  - Dispatches to training, evaluation, config editing, dataset download, and device inspection.
- `configs/`
  - YAML experiment configurations. Prefer adding reproducible runs here.
- `scripts/`
  - Non-interactive helpers (e.g., run with config, dataset download utilities).
- `sustainvision/`
  - Main package with config, training, evaluation, detection, and utility modules.
- `objectdetection/`
  - Legacy/standalone baseline experiments (research scripts, not the main orchestrated pipeline).
- `databases/`
  - Local dataset storage (git-ignored except `.gitkeep`).
- `outputs*/`, `output_*`
  - Experiment artifacts and historical runs (checkpoints, CSVs, logs).

### Python package file-by-file (`sustainvision/`)

- `config.py`
  - Defines `TrainingConfig` and `ConfigManager`.
  - Handles YAML load/save, default merging, and config-path resolution.
  - This is the main compatibility surface for new config options.
- `tui.py`
  - Interactive config editor built with `questionary`.
  - Converts user prompts into config updates and persists them via `ConfigManager`.
- `utils.py`
  - Shared utilities: device resolution, random seeding, report-path helpers.
- `types.py`
  - Shared typed containers/dataclasses for training/evaluation summaries and metrics.
- `data.py`
  - Dataset downloading and loader construction (classification + detection).
  - Contains dataset parsing/normalization/box conversion logic and custom collate functions.
  - Detection dataloaders return `(images, targets)` with standardized target dicts.
- `samplers.py`
  - `MPerClassSampler` used by SupCon workflows to guarantee positives per batch.
- `models.py`
  - Backbone + classifier/projector construction (ResNet, MobileNet, EfficientNet, ViT, fallback MLP).
  - Shared by contrastive/supervised training.
- `losses.py`
  - Standard supervised losses plus SimCLR/SupCon implementations and dispatch helpers.
- `optimizers.py`
  - Optimizer builders (Adam/SGD/etc.) and scheduler construction (step/cosine/warmup).
- `reporting.py`
  - CSV reporting helpers for training metrics and run summaries.
- `training.py`
  - Main training loop implementation for classification/contrastive workflows.
  - Handles checkpoints, CodeCarbon tracking, quantization export, and resource-energy logging.
- `schedule.py`
  - Alternating contrastive pretrain/finetune schedule orchestration.
  - Reuses `training.py` phases with explicit state handoff.
- `evaluation.py`
  - Public evaluation entrypoint (`evaluate_with_head`).
  - Classification evaluation implementation plus dispatch wrappers to detection modules.
  - Kept intentionally thin after the detection refactor.
- `detection_models.py`
  - Detection backbones/heads/factories and pretrained-backbone loading helpers.
  - Includes torchvision detector builders and the shared `MultiHeadModel`/`DetectionHead`.
- `detection_metrics.py`
  - IoU, NMS, COCO/VOC metrics, prediction conversion helpers.
  - Used by multiple detection loops to avoid duplicated metric code.
- `detection_train.py`
  - Detection training/evaluation loop implementations (slot-based and torchvision detector paths).
  - Includes CodeCarbon + `tqdm` integration, CSV/checkpoint saving, and resource logging.
- `rcnn_classic.py`
  - Experimental classic R-CNN-style pipeline:
    - proposal generation
    - ROI crops via `roi_align`
    - backbone feature extraction on ROI patches
    - per-ROI classification + box regression
  - Designed for feature-space probing / experimentation, not as a production detector.

### How the main execution paths fit together

- **Training path**:
  - `main.py` -> `ConfigManager.load()` -> `train_model()` (`training.py`)
  - Optional contrastive schedule -> `schedule.py`
  - Uses `data.py`, `models.py`, `losses.py`, `optimizers.py`, `reporting.py`
- **Evaluation path**:
  - `main.py` -> `evaluate_with_head()` (`evaluation.py`)
  - `classification` head stays in `evaluation.py`
  - detection heads dispatch to:
    - `detection_train.py` (slot / torchvision detectors)
    - `rcnn_classic.py` (experimental classic R-CNN path)
  - Detection modules reuse `detection_models.py`, `detection_metrics.py`, and `data.py`

### Contributor tips (where to change what)

- Add a new config option:
  - `sustainvision/config.py` (defaults + serialization)
  - `sustainvision/tui.py` (prompt, if user-facing)
  - relevant consumer module (`training.py`, `detection_train.py`, `rcnn_classic.py`, etc.)
- Add a new detection metric:
  - implement in `sustainvision/detection_metrics.py`
  - wire into `detection_train.py` and/or `rcnn_classic.py`
- Add a new detection model variant:
  - factory/build logic in `sustainvision/detection_models.py`
  - evaluation dispatch in `sustainvision/evaluation.py`
  - training loop integration in `sustainvision/detection_train.py`
- Add a new experiment:
  - create a YAML in `configs/`
  - ensure `save_model_path` / `report_filename` are unique and meaningful

Latest Changes: `rcnn_classic` Fixes & Improvements
---------------------------------------------------

The `rcnn_classic` path received a substantial quality-of-life and reliability
upgrade. These changes are especially relevant if you use it to probe frozen
feature spaces (e.g., SupCon-pretrained CIFAR backbones transferred to VOC).

### What was fixed

- **Variable-size detection batch crash fixed**
  - The classic R-CNN loop previously assumed `images` was a single stacked tensor.
  - Detection dataloaders may return a list when image sizes differ.
  - The classic path now requests fixed-size resized images from `data.py`, which keeps
    ROI-align inputs batched and consistent.

- **`tqdm` progress bars added**
  - Train/validation loops now use `tqdm` when available (and fall back gracefully if not installed).

- **CodeCarbon support added to `rcnn_classic`**
  - Emissions and energy consumption are now tracked (best effort, optional dependency).
  - Values are returned in evaluation results and can be persisted in artifacts.

- **Artifact saving added (matching other detection loops)**
  - Config snapshot (`config.yaml`)
  - Checkpoints (`epoch_XXXX.pt`, `best_model.pt`)
  - Final results (`results.json`, `results.yaml`)
  - Per-epoch metrics CSV (`detection_metrics.csv`)
  - Resource/energy snapshot CSV (`resource_energy_log.csv`)
  - Final validation predictions and GT dumps for debugging metric mismatches

- **Finish date/time recorded in CSV**
  - `detection_metrics.csv` now includes a summary row with finish timestamps
    (ISO timestamp and local date/time string).

- **CIFAR-pretrained ResNet stem compatibility improved**
  - Added `hyperparameters.backbone_image_size` for `rcnn_classic`.
  - This is important when loading CIFAR-trained checkpoints (e.g., SupCon ResNet18)
    so the backbone stem shape matches and `conv1` weights do not get skipped.

- **Selective-search proposal generation with on-disk caching**
  - New proposal method for `rcnn_classic`:
    - `evaluation.rcnn_classic_proposal_method: selective_search`
  - Proposals can be cached to disk and reused across epochs/reruns, making later runs faster.
  - If OpenCV selective search is unavailable, the code falls back to grid proposals with a warning.

### New/important `rcnn_classic` config knobs

Example (feature-space probe, frozen backbone):

```yaml
evaluation:
  head_type: rcnn_classic
  freeze_backbone: true
  rcnn_classic_proposal_method: selective_search
  rcnn_classic_cache_proposals: true
  rcnn_classic_selective_search_mode: fast
  rcnn_classic_selective_search_min_box_size: 8
  rcnn_classic_max_grid_proposals: 256  # also used as proposal cap for selective search

hyperparameters:
  backbone_image_size: 32  # for CIFAR-trained ResNet stems
  checkpoint_interval: 6
  track_emissions: true
```

### Performance notes (important)

- The **first epoch is slower** when proposal caching is enabled because selective-search
  proposals are computed and written to disk.
- **Later epochs and reruns are faster** because cached proposal tensors are loaded.
- `rcnn_classic_max_grid_proposals` currently acts as the **proposal cap** for both grid
  proposals and selective-search proposals (the name is legacy).
- If GPU memory is tight, reduce:
  - `hyperparameters.batch_size`
  - `evaluation.rcnn_classic_max_grid_proposals`
  - `evaluation.rcnn_classic_roi_size`

### Limitations (current experimental status)

- `rcnn_classic` remains an **experimental feature-space probing path**, not a production detector.
- Lower validation loss does **not** necessarily imply good `map50`; AP depends on proposal
  recall, class confidence ranking, and IoU quality at inference time.
- For strong detection baselines, prefer `torchvision_frcnn` or `torchvision_rcnn`.

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
