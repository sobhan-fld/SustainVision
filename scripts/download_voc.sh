#!/bin/bash
# Script to download Pascal VOC 2012 dataset manually

VOC_DIR="databases/voc"
mkdir -p "$VOC_DIR"
cd "$VOC_DIR"

echo "Downloading Pascal VOC 2012 dataset..."
echo "This will download ~2GB of data and may take 10-30 minutes."

# Try alternative download methods
echo "Attempting download from official source..."

# Method 1: Try with curl (better handling of redirects)
if command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -L -o VOCtrainval_11-May-2012.tar \
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" \
        --progress-bar
else
    # Method 2: Use wget with proper user agent
    echo "Using wget..."
    wget --user-agent="Mozilla/5.0" \
         "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" \
         -O VOCtrainval_11-May-2012.tar
fi

# Check if download was successful (file should be > 1GB)
if [ -f "VOCtrainval_11-May-2012.tar" ]; then
    FILE_SIZE=$(stat -f%z "VOCtrainval_11-May-2012.tar" 2>/dev/null || stat -c%s "VOCtrainval_11-May-2012.tar" 2>/dev/null)
    if [ "$FILE_SIZE" -gt 1000000000 ]; then
        echo "Download successful! Extracting..."
        tar -xf VOCtrainval_11-May-2012.tar
        echo "Extraction complete!"
        echo "Dataset structure: databases/voc/VOCdevkit/VOC2012/"
    else
        echo "ERROR: Downloaded file is too small ($FILE_SIZE bytes)."
        echo "The download may have failed. Please try:"
        echo "  1. Download manually from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
        echo "  2. Or use: python -c \"from torchvision.datasets import VOCDetection; VOCDetection('databases/voc', year='2012', image_set='train', download=True)\""
        exit 1
    fi
else
    echo "ERROR: Download failed."
    exit 1
fi

