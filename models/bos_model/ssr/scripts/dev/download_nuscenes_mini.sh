#!/bin/bash

# Set the download directory
DOWNLOAD_DIR="/home/bos/work/bhrc_user/data/sets"

# Create necessary directories
mkdir -p "$DOWNLOAD_DIR/nuScenes"

# Download and extract v1.0-mini.tgz into nuScenes/
echo "Downloading v1.0-mini.tgz..."
if wget -q --show-progress -O "$DOWNLOAD_DIR/nuscenes/v1.0-mini.tgz" https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz; then
    echo "Extracting v1.0-mini.tgz..."
    tar -xzf "$DOWNLOAD_DIR/nuscenes/v1.0-mini.tgz" -C "$DOWNLOAD_DIR/nuscenes"
    echo "v1.0-mini.tgz downloaded and extracted successfully."
else
    echo "Failed to download v1.0-mini.tgz!" >&2
    exit 1
fi

# Download and extract can_bus.zip into can_bus/
echo "Downloading can_bus.zip..."
if wget -q --show-progress -O "$DOWNLOAD_DIR/can_bus.zip" https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip; then
    echo "Extracting can_bus.zip..."
    unzip -q "$DOWNLOAD_DIR/can_bus.zip" -d "$DOWNLOAD_DIR/can_bus"
    echo "can_bus.zip downloaded and extracted successfully."
else
    echo "Failed to download can_bus.zip!" >&2
    exit 1
fi

# Download and extract nuScenes-map-expansion-v1.3.zip into nuscenes/
echo "Downloading nuScenes-map-expansion-v1.3.zip..."
if wget -q --show-progress -O "$DOWNLOAD_DIR/nuscenes/nuScenes-map-expansion-v1.3.zip" https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip; then
    echo "Extracting nuScenes-map-expansion-v1.3.zip..."
    unzip -q "$DOWNLOAD_DIR/nuscenes/nuScenes-map-expansion-v1.3.zip" -d "$DOWNLOAD_DIR/nuscenes"
    echo "nuScenes-map-expansion-v1.3.zip downloaded and extracted successfully."
else
    echo "Failed to download nuScenes-map-expansion-v1.3.zip!" >&2
    exit 1
fi

echo "All files downloaded and extracted successfully!"