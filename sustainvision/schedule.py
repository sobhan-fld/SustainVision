"""Contrastive learning alternating schedule implementation.

This module handles the alternating pretrain/finetune cycle schedule
for contrastive learning (SimCLR, SupCon, etc.).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .config import TrainingConfig
from .types import EpochMetrics, TrainingRunSummary
from .training import _execute_training_phase
from .utils import unique_report_path
from .reporting import write_report_csv

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None  # type: ignore


def _parse_schedule_config(config: TrainingConfig) -> Dict[str, Any]:
    """Extract and parse schedule configuration parameters."""
    schedule_cfg = config.simclr_schedule or {}
    base_lr = float(config.hyperparameters.get("lr", 1e-3))
    finetune_lr = schedule_cfg.get("finetune_lr")
    
    return {
        "cycles": int(schedule_cfg.get("cycles", 8)),
        "pretrain_epochs": int(schedule_cfg.get("pretrain_epochs", 50)),
        "finetune_epochs": int(schedule_cfg.get("finetune_epochs", 20)),
        "pretrain_loss": str(schedule_cfg.get("pretrain_loss", "simclr")),
        "finetune_loss": str(schedule_cfg.get("finetune_loss", "cross_entropy")),
        "finetune_lr": float(finetune_lr) if finetune_lr is not None else base_lr,
        "freeze_backbone": bool(schedule_cfg.get("freeze_backbone", False)),
        "optimizer_reset": bool(schedule_cfg.get("optimizer_reset", True)),
    }


def _print_schedule_info(params: Dict[str, Any]) -> None:
    """Print schedule configuration information."""
    print(f"\n[info] Starting contrastive learning alternating schedule:")
    print(f"  - Cycles: {params['cycles']}")
    print(f"  - Pretrain: {params['pretrain_epochs']} epochs with {params['pretrain_loss']}")
    print(f"  - Finetune: {params['finetune_epochs']} epochs with {params['finetune_loss']}")
    print(f"  - Finetune LR: {params['finetune_lr']}")
    print(f"  - Freeze backbone: {params['freeze_backbone']}")
    print(f"  - Reset optimizer: {params['optimizer_reset']}\n")


def _setup_emissions_tracker(report_path: Path) -> Any:
    """Initialize and start emissions tracker for the entire schedule run."""
    if EmissionsTracker is None:
        raise RuntimeError("CodeCarbon is required for emissions tracking")
    
    logging.getLogger("codecarbon").setLevel(logging.WARNING)
    tracker = EmissionsTracker(
        measure_power_secs=5,
        project_name=report_path.stem,
        log_level="warning",
    )  # type: ignore[call-arg]
    tracker.start()
    return tracker


def _adjust_epoch_numbers(
    metrics: List[EpochMetrics],
    start_epoch: int,
) -> None:
    """Adjust epoch numbers in metrics to be continuous starting from start_epoch."""
    for metric in metrics:
        metric.epoch = start_epoch + metric.epoch - 1


def _run_pretrain_phase(
    config: TrainingConfig,
    cycle: int,
    params: Dict[str, Any],
    model_state: Optional[dict],
    project_dir: Path,
    report_path: Path,
) -> Tuple[TrainingRunSummary, dict]:
    """Run a single pretrain phase (contrastive learning)."""
    pretrain_label = f"pretrain_cycle_{cycle}"
    print(f"[info] Starting pretrain phase: {pretrain_label}")
    
    summary, state = _execute_training_phase(
        config,
        project_root=project_dir,
        report_path=report_path,
        phase_label=pretrain_label,
        initial_state=model_state,
        write_outputs=False,
        loss_function_override=params["pretrain_loss"],
        epochs_override=params["pretrain_epochs"],
        lr_override=None,  # Use base LR from config
        freeze_backbone_override=False,
        skip_emissions_tracker=True,
    )
    return summary, state


def _run_finetune_phase(
    config: TrainingConfig,
    cycle: int,
    params: Dict[str, Any],
    model_state: Optional[dict],
    project_dir: Path,
    report_path: Path,
) -> Tuple[TrainingRunSummary, dict]:
    """Run a single finetune phase (supervised learning)."""
    finetune_label = f"finetune_cycle_{cycle}"
    print(f"[info] Starting finetune phase: {finetune_label}")
    
    summary, state = _execute_training_phase(
        config,
        project_root=project_dir,
        report_path=report_path,
        phase_label=finetune_label,
        initial_state=model_state,
        write_outputs=False,
        loss_function_override=params["finetune_loss"],
        epochs_override=params["finetune_epochs"],
        lr_override=params["finetune_lr"],
        freeze_backbone_override=params["freeze_backbone"],
        skip_emissions_tracker=True,
        reset_classifier=params["freeze_backbone"],
    )
    return summary, state


def _save_cycle_checkpoint(
    config: TrainingConfig,
    cycle: int,
    model_state: dict,
    optimizer_state: dict,
    report_path: Path,
) -> None:
    """Save model checkpoint after a finetune phase."""
    if torch is None:
        raise RuntimeError("PyTorch is required for checkpoint saving")
    
    artifact_dir = Path(config.save_model_path or "artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{report_path.stem}_cycle{cycle}.pt"
    
    torch.save(
        {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "config": asdict(config),
            "cycle": cycle,
        },
        model_path,
    )
    
    # Save config snapshot
    config_yaml_path = model_path.with_suffix('.yaml')
    with config_yaml_path.open('w', encoding='utf-8') as cfg_handle:
        yaml.safe_dump(asdict(config), cfg_handle, sort_keys=False)
    
    print(f"[info] Checkpoint saved: {model_path}")
    print(f"[info] Config snapshot: {config_yaml_path}")


def _run_single_cycle(
    config: TrainingConfig,
    cycle: int,
    total_cycles: int,
    params: Dict[str, Any],
    model_state: Optional[dict],
    all_epoch_metrics: List[EpochMetrics],
    project_dir: Path,
    report_path: Path,
) -> dict:
    """Run one complete cycle: pretrain + finetune.
    
    Returns:
        Updated model state dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Cycle {cycle}/{total_cycles}")
    print(f"{'='*60}\n")
    
    # Run pretrain phase
    pretrain_summary, pretrain_state = _run_pretrain_phase(
        config, cycle, params, model_state, project_dir, report_path
    )
    model_state = pretrain_state.get("model_state")
    
    # Adjust epoch numbers and accumulate metrics
    current_epoch_start = len(all_epoch_metrics) + 1
    _adjust_epoch_numbers(pretrain_summary.epochs, current_epoch_start)
    all_epoch_metrics.extend(pretrain_summary.epochs)
    
    # Run finetune phase
    finetune_summary, finetune_state = _run_finetune_phase(
        config, cycle, params, model_state, project_dir, report_path
    )
    model_state = finetune_state.get("model_state")
    
    # Adjust epoch numbers and accumulate metrics
    current_epoch_start = len(all_epoch_metrics) + 1
    _adjust_epoch_numbers(finetune_summary.epochs, current_epoch_start)
    all_epoch_metrics.extend(finetune_summary.epochs)
    
    # Save checkpoint after finetune
    if config.save_model:
        _save_cycle_checkpoint(
            config, cycle, model_state, finetune_state.get("optimizer_state", {}), report_path
        )
    
    # Write accumulated metrics to CSV
    write_report_csv(
        report_path,
        all_epoch_metrics,
        emissions_kg=None,  # Will write at end
        energy_kwh=None,
        duration_seconds=None,
        config=config,
        append=False,
    )
    
    return model_state


def run_contrastive_schedule(
    config: TrainingConfig,
    *,
    project_root: Optional[Path] = None,
) -> TrainingRunSummary:
    """Execute the full contrastive learning alternating schedule.
    
    Runs multiple cycles, each consisting of:
    1. Pretrain phase (contrastive learning)
    2. Finetune phase (supervised learning)
    
    Args:
        config: Training configuration with simclr_schedule enabled
        project_root: Root directory for outputs (defaults to current directory)
    
    Returns:
        TrainingRunSummary with aggregated metrics and emissions
    """
    # Setup paths and parse configuration
    project_dir = project_root or Path.cwd()
    report_path = unique_report_path(project_dir, config.report_filename)
    params = _parse_schedule_config(config)
    
    # Print schedule information
    _print_schedule_info(params)
    
    # Initialize tracking variables
    all_epoch_metrics: List[EpochMetrics] = []
    model_state: Optional[dict] = None
    
    # Setup emissions tracker for entire run
    tracker = _setup_emissions_tracker(report_path)
    start_time = time.perf_counter()
    
    try:
        # Run all cycles
        for cycle in range(1, params["cycles"] + 1):
            model_state = _run_single_cycle(
                config,
                cycle,
                params["cycles"],
                params,
                model_state,
                all_epoch_metrics,
                project_dir,
                report_path,
            )
    
    finally:
        # Stop tracker and collect emissions data
        total_emissions_kg = tracker.stop()
        emissions_data = getattr(tracker, "final_emissions_data", None)
        total_duration = time.perf_counter() - start_time
        
        total_energy_kwh = None
        if emissions_data is not None:
            total_energy_kwh = getattr(emissions_data, "energy_consumed", None)
            if hasattr(emissions_data, "duration") and emissions_data.duration is not None:
                total_duration = emissions_data.duration
    
    # Write final summary row with emissions
    write_report_csv(
        report_path,
        all_epoch_metrics,
        emissions_kg=total_emissions_kg,
        energy_kwh=total_energy_kwh,
        duration_seconds=total_duration,
        config=config,
        append=False,
    )
    
    # Print completion summary
    print(f"\n{'='*60}")
    print(f"Contrastive learning schedule complete!")
    print(f"Total cycles: {params['cycles']}")
    print(f"Total epochs: {len(all_epoch_metrics)}")
    print(f"Report: {report_path}")
    print(f"{'='*60}\n")
    
    return TrainingRunSummary(
        report_path=report_path,
        emissions_kg=total_emissions_kg,
        energy_kwh=total_energy_kwh,
        duration_seconds=total_duration,
        epochs=all_epoch_metrics,
    )

