import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.nn.utils.rnn import pad_sequence  # Import pad_sequence
from qa_model import QA_Model

# Load data from JSON file
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Preprocessing function to convert text to indices
def preprocess_data(data, vocab):
    questions = []
    answers = []
    for item in data:
        questions.append(torch.tensor([vocab[word] for word in item['question'].lower().split()], dtype=torch.long))  # Convert to tensor
        answers.append(torch.tensor([vocab[word] for word in item['answer'].lower().split()], dtype=torch.long))      # Convert to tensor
    return questions, answers

# Creating a simple vocabulary
def create_vocab(data):
    vocab = {}
    for item in data:
        for word in item['question'].lower().split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Start indexing from 1
        for word in item['answer'].lower().split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1
    return vocab

# Training the model
def train_model(model, data, num_epochs=100):
    questions, answers = preprocess_data(data, vocab)
    
    # Pad sequences to ensure consistent shape
    input_tensor = pad_sequence(questions, batch_first=True)  # Pad questions
    target_tensor = pad_sequence(answers, batch_first=True)    # Pad answers

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start timing the training
    start_time = time.time()  # Record the start time

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(input_tensor)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_tensor.view(-1))  # Flatten target tensor for loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Calculate training time
    end_time = time.time()  # Record the end time
    training_time = end_time - start_time  # Calculate the total training time
    print(f'Training completed in {training_time:.2f} seconds.')

# Load data from the JSON file
data = load_data_from_json('data.json')

# Create the vocabulary from loaded data
vocab = create_vocab(data)

# Create the model
vocab_size = len(vocab) + 1  # Plus one for padding
embedding_dim = 16
hidden_dim = 32
model = QA_Model(vocab_size, embedding_dim, hidden_dim)

# Train the model
train_model(model, data)

# Export the model to ONNX
dummy_input = torch.randint(0, vocab_size, (1, 10))  # Batch size of 1, sequence length of 10
torch.onnx.export(model, dummy_input, '../qa_model.onnx', export_params=True, opset_version=11,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
