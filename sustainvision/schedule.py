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
        "finetune_optimizer": schedule_cfg.get("finetune_optimizer"),
        "finetune_weight_decay": (
            float(schedule_cfg.get("finetune_weight_decay"))
            if schedule_cfg.get("finetune_weight_decay") is not None
            else None
        ),
        "freeze_backbone": bool(schedule_cfg.get("freeze_backbone", False)),
        "optimizer_reset": bool(schedule_cfg.get("optimizer_reset", True)),
        "use_reference_transforms": bool(schedule_cfg.get("use_reference_transforms", False)),
        "linear_subset_per_class": schedule_cfg.get("linear_subset_per_class"),
        "linear_subset_seed": schedule_cfg.get("linear_subset_seed", config.seed),
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
    scheduler_state: Optional[dict] = None,
    cumulative_scheduler_steps: Optional[int] = None,
    total_t_max: Optional[int] = None,
    total_cycles: Optional[int] = None,
) -> Tuple[TrainingRunSummary, dict]:
    """Run a single pretrain phase (contrastive learning).
    
    Always runs pretrain with unfrozen backbone. This allows the backbone
    to continue improving across cycles, while periodic finetune phases
    evaluate the backbone quality with a fresh linear head.
    
    The scheduler continues across pretrain cycles to enable continued learning.
    """
    pretrain_label = f"pretrain_cycle_{cycle}"
    print(f"[info] Starting pretrain phase: {pretrain_label} (backbone unfrozen)")
    if cumulative_scheduler_steps is not None:
        print(f"[info] Continuing scheduler from step {cumulative_scheduler_steps}")
    
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
        freeze_backbone_override=False,  # Always unfrozen during pretrain
        skip_emissions_tracker=False,
        scheduler_config_override=None,
        simclr_recipe_override=params["use_reference_transforms"],
        scheduler_state=scheduler_state,
        cumulative_scheduler_steps=cumulative_scheduler_steps,
        total_t_max=total_t_max,  # Pass through total_t_max for consistent scheduling
        total_cycles=total_cycles,  # Pass total_cycles to calculate total_t_max in first cycle
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
    """Run a single finetune phase (supervised learning).
    
    This phase evaluates the backbone quality by training a fresh linear head
    with frozen backbone. After this phase, the backbone will be unfrozen again
    in the next pretrain phase to continue improving.
    """
    finetune_label = f"finetune_cycle_{cycle}"
    print(f"[info] Starting finetune phase: {finetune_label} (backbone frozen, evaluating representation quality)")
    
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
        optimizer_override=params.get("finetune_optimizer"),
        weight_decay_override=params.get("finetune_weight_decay"),
        freeze_backbone_override=params["freeze_backbone"],
        skip_emissions_tracker=False,
        reset_classifier=params["freeze_backbone"],  # Reset classifier to evaluate backbone quality
        scheduler_config_override=None,
        simclr_recipe_override=params["use_reference_transforms"],
        subset_per_class_override=params.get("linear_subset_per_class"),
        subset_seed_override=params.get("linear_subset_seed"),
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
    scheduler_state: Optional[dict] = None,
    cumulative_scheduler_steps: Optional[int] = None,
    total_t_max: Optional[int] = None,
) -> Tuple[dict, Optional[int], Optional[dict], Optional[int]]:
    """Run one complete cycle: pretrain + finetune.
    
    Returns:
        Tuple of (updated model state dictionary, updated cumulative_scheduler_steps, 
                 updated scheduler_state, emissions_kg, energy_kwh) for this cycle.
    """
    print(f"\n{'='*60}")
    print(f"Cycle {cycle}/{total_cycles}")
    print(f"{'='*60}\n")
    
    # Run pretrain phase (scheduler continues across pretrain cycles)
    # Pass total_t_max and total_cycles to ensure consistent scheduler behavior
    pretrain_summary, pretrain_state = _run_pretrain_phase(
        config, cycle, params, model_state, project_dir, report_path,
        scheduler_state=scheduler_state,
        cumulative_scheduler_steps=cumulative_scheduler_steps,
        total_t_max=total_t_max,
        total_cycles=total_cycles,
    )
    model_state = pretrain_state.get("model_state")
    scheduler_state = pretrain_state.get("scheduler_state")
    # Update total_t_max from pretrain state (calculated in first cycle)
    total_t_max = pretrain_state.get("total_t_max", total_t_max)
    
    # Update cumulative steps from scheduler state
    # The scheduler state contains 'last_epoch' which is the total steps taken
    if scheduler_state is not None:
        # For CosineAnnealingLR, state_dict contains 'last_epoch' which is the step count
        # For Sequential schedulers (warmup_cosine), we need to check the last scheduler
        if isinstance(scheduler_state, dict):
            # Try to get last_epoch from the state
            if "last_epoch" in scheduler_state:
                cumulative_scheduler_steps = scheduler_state["last_epoch"]
            elif "schedulers" in scheduler_state:
                # Sequential scheduler - get last_epoch from the last scheduler
                last_sched_state = scheduler_state["schedulers"][-1] if scheduler_state["schedulers"] else {}
                cumulative_scheduler_steps = last_sched_state.get("last_epoch", cumulative_scheduler_steps or 0)
    # If no scheduler state, cumulative_steps remains as passed (None or previous value)
    
    # Adjust epoch numbers and accumulate metrics
    current_epoch_start = len(all_epoch_metrics) + 1
    _adjust_epoch_numbers(pretrain_summary.epochs, current_epoch_start)
    all_epoch_metrics.extend(pretrain_summary.epochs)
    
    # Run finetune phase (scheduler resets for finetune - uses different LR)
    finetune_summary, finetune_state = _run_finetune_phase(
        config, cycle, params, model_state, project_dir, report_path
    )
    model_state = finetune_state.get("model_state")
    # Don't update scheduler_state from finetune - it uses a different scheduler
    
    # Adjust epoch numbers and accumulate metrics
    current_epoch_start = len(all_epoch_metrics) + 1
    _adjust_epoch_numbers(finetune_summary.epochs, current_epoch_start)
    all_epoch_metrics.extend(finetune_summary.epochs)

    # Collect emissions/energy for this cycle from both phases (if available)
    cycle_emissions_kg: Optional[float] = None
    cycle_energy_kwh: Optional[float] = None
    for summary in (pretrain_summary, finetune_summary):
        if summary.emissions_kg is not None:
            cycle_emissions_kg = (cycle_emissions_kg or 0.0) + float(summary.emissions_kg)
        if summary.energy_kwh is not None:
            cycle_energy_kwh = (cycle_energy_kwh or 0.0) + float(summary.energy_kwh)
    
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
    
    return model_state, cumulative_scheduler_steps, scheduler_state, total_t_max, cycle_emissions_kg, cycle_energy_kwh


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
    project_dir = project_root or (Path.cwd() / "outputs")
    project_dir.mkdir(parents=True, exist_ok=True)
    report_path = unique_report_path(project_dir, config.report_filename)
    params = _parse_schedule_config(config)
    
    # Print schedule information
    _print_schedule_info(params)
    
    # Initialize tracking variables
    all_epoch_metrics: List[EpochMetrics] = []
    model_state: Optional[dict] = None
    scheduler_state: Optional[dict] = None
    cumulative_scheduler_steps: Optional[int] = None
    total_t_max: Optional[int] = None
    total_emissions_kg: Optional[float] = None
    total_energy_kwh: Optional[float] = None
    start_time = time.perf_counter()
    
    # Run all cycles
    for cycle in range(1, params["cycles"] + 1):
        model_state, cumulative_scheduler_steps, scheduler_state, total_t_max, cycle_emissions_kg, cycle_energy_kwh = _run_single_cycle(
            config,
            cycle,
            params["cycles"],
            params,
            model_state,
            all_epoch_metrics,
            project_dir,
            report_path,
            scheduler_state=scheduler_state,
            cumulative_scheduler_steps=cumulative_scheduler_steps,
            total_t_max=total_t_max,
        )
        if cycle_emissions_kg is not None:
            total_emissions_kg = (total_emissions_kg or 0.0) + float(cycle_emissions_kg)
        if cycle_energy_kwh is not None:
            total_energy_kwh = (total_energy_kwh or 0.0) + float(cycle_energy_kwh)
    
    total_duration = time.perf_counter() - start_time
    
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

