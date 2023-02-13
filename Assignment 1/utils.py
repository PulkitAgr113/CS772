import json,torch
from torch.utils.data import Dataset
import numpy as np
from back_prop import Model

# class Pentagram(Dataset):
#     def __init__(self, train_sents):
#         pentagrams = []
#         for line in train_sents:
#             words = line.strip().split()
#             for ind in range(2, len(words)-2):
#                 pentagrams.append([words[ind-2], words[ind-1], words[ind], words[ind+1], words[ind+2]])

#         self.pentagrams = pentagrams

#     def __len__(self):
#         return len(self.pentagrams)
    
#     def __getitem__(self, idx):
#         return self.pentagrams[idx]

class nGram(Dataset):
    def __init__(self, train_sents, n):
        ngrams = []
        for line in train_sents:
            words = line.strip().split()
            for ind in range(n, len(words)-n):
                ls = []
                for x in range(ind-n, ind+n+1):
                    ls.append(words[x])

                ngrams.append(ls)

        self.ngrams = ngrams

    def __len__(self):
        return len(self.ngrams)
    
    def __getitem__(self, idx):
        return self.ngrams[idx]
    
def changed_words():
    return {"Gujurat": "Gujarat", "Telengana": "Telangana",
            "Maharastra": "Maharashtra", "Slovakian": "Slovaks",
            "Kiev" : "Kyiv", "USA": "United States", "Bengaluru": "Bangalore"}

def vocab():
    with open("all_vocab.json", 'r') as file:
        vocab_dict = json.load(file)
    return vocab_dict

def vocab_inv():
    vocab_dict_inv = {}
    vocab_dict = vocab()
    for word in vocab_dict:
        id = vocab_dict[word]
        vocab_dict_inv[id] = word

    return vocab_dict_inv

def get_one_hot_encoding(vocab_dict, words):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab_dict)
    inds = np.array(list(map(lambda x: int(vocab_dict[x]), words)))
    vecs = np.zeros((inds.size, vocab_size))
    vecs[np.arange(inds.size), inds] = 1
    vecs = torch.from_numpy(vecs).float().to(device)
    return vecs

def params(file_name):
    vocab_dict = vocab()
    embedding_dim = 700
    lr = 0.01
    epochs = 1
    batch_size = 128
    model = Model(len(vocab_dict), embedding_dim, lr)

    with open(file_name, 'r') as file:
        train_sents = file.readlines()

    return epochs, model, batch_size, vocab_dict, train_sents

    

