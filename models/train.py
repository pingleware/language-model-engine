# train.py

import torch
import numpy as np
from model import create_model

def export_to_onnx(model_path='model.onnx'):
    # Create a model instance
    model = create_model()

    # Dummy input for tracing the model
    dummy_input = torch.randn(1, 10)  # Batch size of 1 and input size of 10
    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, model_path, export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    export_to_onnx()
