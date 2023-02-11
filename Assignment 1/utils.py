import json
from torch.utils.data import Dataset
import numpy as np
from back_prop import Model

class Pentagram(Dataset):
    def __init__(self, train_sents):
        pentagrams = []
        for line in train_sents:
            words = line.strip().split()
            for ind in range(2, len(words)-2):
                pentagrams.append([words[ind-2], words[ind-1], words[ind], words[ind+1], words[ind+2]])

        self.pentagrams = pentagrams

    def __len__(self):
        return len(self.pentagrams)
    
    def __getitem__(self, idx):
        return self.pentagrams[idx]

def vocab():
    with open("vocab.json", 'r') as file:
        vocab_dict = json.load(file)
    return vocab_dict

def get_one_hot_encoding(vocab_dict, words):
    vocab_size = len(vocab_dict)
    inds = np.array(list(map(lambda x: int(vocab_dict[x]), words)))
    vecs = np.zeros((inds.size, vocab_size))
    vecs[np.arange(inds.size), inds] = 1
    return vecs

def params():
    vocab_dict = vocab()
    embedding_dim = 100
    lr = 0.1
    epochs = 2
    batch_size = 128
    model = Model(len(vocab_dict), embedding_dim, lr)

    with open("training_sents.txt", 'r') as file:
        train_sents = file.readlines()

    return epochs, model, batch_size, vocab_dict, train_sents

    

