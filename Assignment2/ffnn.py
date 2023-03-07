# from gensim.models import KeyedVectors
import torch
from torch import nn

# word_vectors_google = KeyedVectors.load('vectors.bin')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.softmax(out)
        return out
    
input_size = 300
hidden_size = 64
num_classes = 12
epochs = 2
lr = 0.001

model = NeuralNet(input_size, hidden_size, num_classes)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

