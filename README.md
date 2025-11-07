SustainVision TUI and Config Manager
====================================

This project provides a minimal TUI to configure a training job and a YAML-backed
configuration manager. You can choose model, dataset, device (CPU or specific CUDA
index), and basic hyperparameters. The configuration is persisted to:

- Linux: `~/.config/sustainvision/config.yaml` (or `$XDG_CONFIG_HOME/sustainvision/config.yaml`)

Install dependencies
--------------------

```bash
pip install -r requirements.txt
```

- PyTorch is optional and only used to detect CUDA devices. If not installed, the
  device list will include only `cpu`.

Run the TUI
-----------

```bash
python main.py
```

After saving, the config is printed and stored on disk. You can import
`ConfigManager` from `sustainvision.config` to programmatically load/edit/save
the configuration in other scripts.

Next steps (not yet implemented)
--------------------------------

- Integrate a training loop
- Measure carbon emissions via `codecarbon`
- Add model/database presets as needed

