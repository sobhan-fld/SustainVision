#!/usr/bin/env python3
"""Download Pascal VOC 2012 dataset using torchvision."""

import sys
from pathlib import Path

try:
    from torchvision.datasets import VOCDetection
except ImportError:
    print("ERROR: torchvision is not installed.")
    print("Install it with: pip install torchvision")
    sys.exit(1)

voc_dir = Path(__file__).parent.parent / "databases" / "voc"
voc_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading Pascal VOC 2012 to: {voc_dir}")
print("This will download ~2GB and may take 10-30 minutes...")
print()

try:
    # This will download and extract automatically
    dataset = VOCDetection(
        root=str(voc_dir),
        year="2012",
        image_set="train",
        download=True,
    )
    print(f"\n✓ Download successful!")
    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Location: {voc_dir}")
    print(f"  Expected structure: {voc_dir}/VOCdevkit/VOC2012/")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nAlternative: Download manually from:")
    print("  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
    print(f"  Extract to: {voc_dir}")
    sys.exit(1)

