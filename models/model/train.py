# train.py
import torch
import torch.optim as optim
import torch.nn as nn 
import numpy as np
from model import create_model

def export_to_onnx(model_path='../model.onnx'):
    # Create a model instance
    model = create_model()

    # Set the model to training mode
    model.train()

    # Dummy training data
    num_samples = 100  # Number of samples
    input_data = torch.randn(num_samples, 10)  # 100 samples, input size 10
    target_data = torch.randn(num_samples, 1)   # 100 samples, output size 1

    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()   # Zero the gradients
        outputs = model(input_data)  # Forward pass
        loss = criterion(outputs, target_data)  # Calculate loss
        loss.backward()        # Backward pass
        optimizer.step()       # Update weights

        if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    # Export the model to ONNX
    dummy_input = torch.randn(1, 10)  # Batch size of 1 and input size of 10
    torch.onnx.export(model, dummy_input, model_path, export_params=True, opset_version=11, 
                      do_constant_folding=True,
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    export_to_onnx()
