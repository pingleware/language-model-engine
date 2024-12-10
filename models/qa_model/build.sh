#!/bin/bash

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install torch numpy onnx transformers onnxruntime onnx2pytorch

echo "Virtual environment created and packages installed."

# Run the training script
#python train.py

# Verify the model
#python verify.py

# Run inference
python qa_inference.py

# Convert to pytorch
#python convert.py

# Deactivate
# deactivate

# remove venv directory
#rm -frd venv
#rm -frd __pycache__
