#!/bin/bash

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install torch onnx torchvision torchaudio


# Run the training script
python train.py

# Verify the model
python verify.py

# Deactivate the virtual environment
deactivate

# remove venv directory
rm -frd venv