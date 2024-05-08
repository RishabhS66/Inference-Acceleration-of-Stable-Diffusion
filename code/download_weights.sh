#!/bin/bash

# Define the target directory for storing weights
target_dir="./data/weights"

# Check if the 'data' directory exists
if [ ! -d "./data" ]; then
    echo "Error: Directory 'data' not found."
    exit 1
fi

# Ensure that the 'weights' directory exists or create it
mkdir -p "$target_dir"

# Download the weights file directly into the 'weights' directory
wget -O "$target_dir/v1-5-pruned-emaonly.ckpt" https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Weights file downloaded successfully."
else
    echo "Error: Failed to download weights file."
    exit 1
fi
