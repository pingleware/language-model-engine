# model.py
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Input size of 10, hidden layer of size 50
        self.fc2 = nn.Linear(50, 1)    # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_model():
    return SimpleModel()
