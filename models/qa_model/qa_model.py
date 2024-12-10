import torch
import torch.nn as nn

class QA_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(QA_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output layer for vocab size

    def forward(self, x):
        x = self.embedding(x)  # Convert input to embeddings
        x, _ = self.lstm(x)  # LSTM processing
        x = self.fc(x[:, -1, :])  # Get output from the last time step
        return x

