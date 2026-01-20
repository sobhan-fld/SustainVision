"""SustainVision entrypoint with training menu."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import questionary

from sustainvision.config import ConfigManager
from sustainvision.data import prompt_and_download
from sustainvision.tui import run_config_tui
from sustainvision.training import MissingDependencyError, train_model
from sustainvision.types import TrainingRunSummary
from sustainvision.evaluation import evaluate_with_head


def _start_training(cm: ConfigManager) -> None:
    """Kick off a training run and display summary output."""

    config = cm.load()
    output_root = Path.cwd() / "outputs"
    try:
        summary: TrainingRunSummary = train_model(config, project_root=output_root)
    except MissingDependencyError as exc:
        print(f"\n[error] {exc}\n")
        return
    except Exception as exc:  # pragma: no cover - runtime protection
        print(f"\n[error] Training failed: {exc}\n")
        return

    print("\nTraining summary:")
    print(f"- report: {summary.report_path}")
    if summary.emissions_kg is not None:
        print(f"- emissions (kg CO₂e): {summary.emissions_kg:.6f}")
    if summary.energy_kwh is not None:
        print(f"- energy (kWh): {summary.energy_kwh:.6f}")
    if summary.duration_seconds is not None:
        print(f"- duration (s): {summary.duration_seconds:.2f}")
    if summary.quantized_model_path is not None:
        print(f"- quantized model: {summary.quantized_model_path}")


def _start_evaluation(cm: ConfigManager) -> None:
    """Evaluate a pretrained backbone with different heads."""

    config = cm.load()
    eval_cfg = config.evaluation or {}
    
    if not eval_cfg.get("enabled", False):
        print("\n[error] Evaluation mode is not enabled in config.")
        print("Set evaluation.enabled: true in your config file.\n")
        return
    
    checkpoint_path = eval_cfg.get("checkpoint_path")
    if not checkpoint_path:
        checkpoint_path = config.checkpoint_path
    
    if not checkpoint_path:
        print("\n[error] No checkpoint path specified.")
        print("Set evaluation.checkpoint_path or checkpoint_path in your config file.\n")
        return
    
    head_type = eval_cfg.get("head_type", "classification")
    
    print(f"\n[info] Starting evaluation with {head_type} head")
    print(f"[info] Checkpoint: {checkpoint_path}")
    print(f"[info] Model: {config.model}")
    print(f"[info] Dataset: {config.database}\n")
    
    # For evaluation, treat the **project root** as the current working directory.
    # Dataset paths (e.g. `databases/coco`) are resolved relative to this root.
    project_root = Path.cwd()
    
    try:
        results = evaluate_with_head(
            config,
            checkpoint_path,
            head_type=head_type,
            project_root=project_root,
            num_anchors=eval_cfg.get("num_anchors", 9),
            hidden_dim=eval_cfg.get("hidden_dim", 256),
        )
        
        print("\nEvaluation results:")
        for key, value in results.items():
            print(f"- {key}: {value}")
        
    except Exception as exc:  # pragma: no cover - runtime protection
        print(f"\n[error] Evaluation failed: {exc}\n")
        import traceback
        traceback.print_exc()


def _choose_and_run_config(configs_dir: Path) -> None:
    """Allow user to pick a YAML config from a directory and run it."""

    configs_dir = configs_dir if configs_dir.is_absolute() else Path.cwd() / configs_dir
    if not configs_dir.is_dir():
        print(f"\n[error] Configs directory not found: {configs_dir}\n")
        return

    yaml_files = sorted(list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml")))
    if not yaml_files:
        print(f"\n[error] No YAML configs found in {configs_dir}\n")
        return

    choices = [f"{idx}: {p.relative_to(Path.cwd())}" for idx, p in enumerate(yaml_files)]
    selection = questionary.select(
        "Select a config to run:",
        choices=choices,
    ).ask()

    if selection is None:
        return

    try:
        selected_idx = int(selection.split(":")[0])
    except Exception:
        print("\n[error] Invalid selection.\n")
        return

    if selected_idx < 0 or selected_idx >= len(yaml_files):
        print("\n[error] Selection out of range.\n")
        return

    config_path = str(yaml_files[selected_idx])
    cm = ConfigManager(config_path)
    cfg = cm.load()

    # If evaluation is enabled in this config, run evaluation instead of training.
    eval_cfg = getattr(cfg, "evaluation", {}) or {}
    if eval_cfg.get("enabled", False):
        _start_evaluation(cm)
    else:
        _start_training(cm)
def _show_devices() -> None:
    """Print available CPU/GPU devices and basic utilization info."""

    print("\nDevice inventory:")

    cpu_count = os.cpu_count()
    print(f"- CPU threads: {cpu_count if cpu_count is not None else 'unknown'}")
    if hasattr(os, "getloadavg"):
        try:
            load1, load5, load15 = os.getloadavg()
            print(f"  Current load averages (1m/5m/15m): {load1:.2f} / {load5:.2f} / {load15:.2f}")
        except OSError:
            pass

    try:
        import torch
    except Exception:
        print("- PyTorch not available; skipping GPU inspection.")
        torch = None  # type: ignore

    if torch is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for idx in range(gpu_count):
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total_mem_gb = props.total_memory / (1024 ** 3)
            with torch.cuda.device(idx):
                try:
                    free, total = torch.cuda.mem_get_info()
                    used = total - free
                    free_gb = free / (1024 ** 3)
                    used_gb = used / (1024 ** 3)
                    print(
                        f"- GPU {idx}: {name} | total {total_mem_gb:.2f} GB | "
                        f"used {used_gb:.2f} GB | free {free_gb:.2f} GB"
                    )
                except Exception:
                    print(f"- GPU {idx}: {name} | total {total_mem_gb:.2f} GB")
    else:
        print("- No CUDA devices detected (torch.cuda.is_available() == False).")

    # Try to show nvidia-smi snapshots if available
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            print("\n[nvidia-smi]")
            print(result.stdout.strip())
        elif result.stderr:
            print(f"\n[nvidia-smi warning] {result.stderr.strip()}")
    except FileNotFoundError:
        pass


def main() -> None:
    """Interactive menu for configuring and training SustainVision pipelines."""

    cm = ConfigManager()
    cm.load()

    while True:
        choice = questionary.select(
            "Select an action:",
            choices=[
                "Start training",
                "Evaluate pretrained model",
                "Run config from configs/",
                "Configure settings",
                "Download databases",
                "Inspect devices",
                "Show current config",
                "Exit",
            ],
            default="Start training",
        ).ask()

        if choice == "Configure settings":
            run_config_tui(cm.path)
            cm.load()  # refresh in-memory config after edits
        elif choice == "Start training":
            _start_training(cm)
        elif choice == "Evaluate pretrained model":
            _start_evaluation(cm)
        elif choice == "Run config from configs/":
            _choose_and_run_config(Path("configs"))
        elif choice == "Download databases":
            dataset_path = prompt_and_download(Path.cwd())
            if dataset_path is not None:
                try:
                    relative_path = dataset_path.relative_to(Path.cwd())
                except ValueError:
                    relative_path = dataset_path
                cm.set(database=str(relative_path))
                cm.save()
                print("Configuration updated with new dataset path.")
        elif choice == "Inspect devices":
            _show_devices()
        elif choice == "Show current config":
            cfg = cm.load()
            print("\nCurrent configuration:")
            print(cfg)
        elif choice == "Exit" or choice is None:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()