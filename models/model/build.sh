#!/bin/bash

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

echo "Virtual environment created and packages installed."

# Remove existing mode
rm ../model.onnx

# Run the training script
python train.py

# Verify the model
python verify.py

# remove venv directory
rm -frd venv
rm -frd __pycache__