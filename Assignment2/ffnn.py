from gensim.models import KeyedVectors
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from dataset import data

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

class dataloader(Dataset):
    def __init__(self, folded_train_sents, folded_train_correct_tags, prev_words, tagidxdict, word_vectors_google, input_size, num_classes):
        inp_out = []
        for train_sents, train_correct_tags in zip(folded_train_sents, folded_train_correct_tags):
            for sent, correct_tags in zip(train_sents, train_correct_tags):
                ind = 0
                for ind in range(len(sent)):
                    elem = np.zeros(input_size + num_classes)
                    corr_tag = correct_tags[ind]
                    idx = tagidxdict[corr_tag]
                    elem[input_size+idx] = 1
                    for word in range(max(ind-prev_words, 0), ind+1):
                        if sent[word] in word_vectors_google:
                            elem[:input_size] += word_vectors_google[sent[word]]
                    inp_out.append(elem)
        
        self.inp_out = inp_out
                
    def __len__(self):
        return len(self.inp_out)
    
    def __getitem__(self, idx):
        return self.inp_out[idx]

def run(model, criterion, optimizer, train_sents, train_correct_tags, tagidxdict, word_vectors_google, batch_size, input_size, num_classes, prev_words = 4):
    train_vectors = dataloader(train_sents, train_correct_tags, prev_words, tagidxdict, word_vectors_google, input_size, num_classes)
    train_dataloader = DataLoader(train_vectors, batch_size = batch_size, shuffle=True)
    error = 0
    for idx,item in enumerate(tqdm(train_dataloader)):
        input_vec = item[:input_size]
        output_vec = item[input_size:]
        pred_vec = model.forward(input_vec)
        loss = criterion(pred_vec, output_vec)
        error += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return error

if __name__ == '__main__':
    prev_words = 4
    input_size = 300
    hidden_size = 64
    num_classes = 12
    epochs = 2
    lr = 0.001
    batch_size = 256

    model = NeuralNet(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    word_vectors_google = KeyedVectors.load('vectors.bin')
    print("Word vectors loaded")

    tagidxdict, folded_sentences, folded_correct_tags = data()

    for epoch in range(epochs):
        error = run(model, criterion, optimizer, folded_sentences, folded_correct_tags, tagidxdict, word_vectors_google, batch_size, input_size, num_classes)
        print(f"Epoch {epoch} completed. Error: {error}")
