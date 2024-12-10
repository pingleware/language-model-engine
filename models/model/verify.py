import onnx

# Load the ONNX model
onnx_model = onnx.load("../model.onnx")

# Check that the model is well-formed
onnx.checker.check_model(onnx_model)
print("The model is valid.")