import torch
import onnx
from onnx2pytorch import ConvertModel

# Load ONNX model
onnx_model = onnx.load("../qa_model.onnx")

# Convert ONNX to PyTorch
pytorch_model = ConvertModel(onnx_model)

# Save the PyTorch model
torch.save(pytorch_model.state_dict(), "../qa_model.pth")
