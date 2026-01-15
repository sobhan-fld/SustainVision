#!/usr/bin/env python3
"""Download Pascal VOC 2012 dataset using torchvision."""

import sys
from pathlib import Path

try:
    from torchvision.datasets import VOCDetection
except ImportError:
    print("ERROR: torchvision is not installed.")
    print("Make sure you're in the correct conda/venv environment.")
    print("Install with: pip install torchvision")
    sys.exit(1)

voc_dir = Path("databases/voc")
voc_dir.mkdir(parents=True, exist_ok=True)

# Remove corrupted file if it exists
corrupted = voc_dir / "VOCtrainval_11-May-2012.tar"
if corrupted.exists() and corrupted.stat().st_size < 10000:  # Less than 10KB = corrupted
    print(f"Removing corrupted file: {corrupted}")
    corrupted.unlink()

print(f"Downloading Pascal VOC 2012 to: {voc_dir.absolute()}")
print("This will download ~2GB and may take 10-30 minutes...")
print()

try:
    dataset = VOCDetection(
        root=str(voc_dir),
        year="2012",
        image_set="train",
        download=True,
    )
    print(f"\n✓ Download successful!")
    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Location: {voc_dir.absolute()}")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nAlternative: Download manually from:")
    print("  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
    print(f"  Extract to: {voc_dir.absolute()}")
    sys.exit(1)
