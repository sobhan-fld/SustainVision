"""Custom data samplers for contrastive learning."""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from torch.utils.data.sampler import Sampler
except Exception:
    # Fallback for when PyTorch is not available
    class Sampler:  # type: ignore
        pass


class MPerClassSampler(Sampler):
    """
    At every iteration, samples K classes, and then samples M images for each of those classes.
    Batch size = K * M.
    
    This ensures each batch contains multiple samples per class, which is crucial for
    supervised contrastive learning (SupCon) to have positive pairs in every batch.
    """
    
    def __init__(
        self,
        labels: list | np.ndarray,
        m_per_class: int,
        batch_size: int,
        length_strategy: str = "full_epoch",
        seed: Optional[int] = None,
    ):
        if Sampler is None:
            raise RuntimeError("PyTorch is required for samplers")
        
        self.labels = np.array(labels)
        self.m_per_class = m_per_class
        self.batch_size = batch_size
        self.classes_per_batch = batch_size // m_per_class
        
        if self.classes_per_batch == 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= m_per_class ({m_per_class})"
            )
        
        self.length_strategy = length_strategy
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Organize indices by class
        self.indices_by_class = {}
        unique_classes = np.unique(self.labels)
        for cls in unique_classes:
            self.indices_by_class[cls] = np.where(self.labels == cls)[0]
            
        self.all_classes = list(self.indices_by_class.keys())
        
        # Calculate length
        if length_strategy == "full_epoch":
            # Roughly number of batches to cover all data
            self.num_batches = len(self.labels) // self.batch_size
            if self.num_batches == 0:
                self.num_batches = 1
        else:
            self.num_batches = 100  # custom fixed length

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        # Determine how many batches we need to generate
        batches_generated = 0
        while batches_generated < self.num_batches:
            # 1. Pick random classes for this batch
            # If we don't have enough classes, allow replacement
            replace = len(self.all_classes) < self.classes_per_batch
            selected_classes = self.rng.choice(
                self.all_classes, 
                self.classes_per_batch, 
                replace=replace
            )
            
            batch_indices = []
            
            # 2. For each selected class, pick M random images
            for cls in selected_classes:
                # If we don't have enough images in the class, allow replacement
                replace = len(self.indices_by_class[cls]) < self.m_per_class
                
                selected_indices = self.rng.choice(
                    self.indices_by_class[cls],
                    self.m_per_class,
                    replace=replace
                )
                batch_indices.extend(selected_indices.tolist())
            
            # 3. Shuffle the batch so the model doesn't learn the order
            self.rng.shuffle(batch_indices)
            
            # Yield indices one by one (DataLoader collects them into a batch)
            for idx in batch_indices:
                yield int(idx)
            
            batches_generated += 1

