import pickle
from utils import vocab
from most_similar import cosine_similarity
import torch

def load_weights():
    with open("skipgram_model.pkl",'rb') as f:
        model=pickle.load(f)

    weights=model.fc1.weights
    return weights

weights=load_weights().detach().cpu().numpy()
w1="walk"
w2="walks"
w3="see"
w4="shuffle"
vocab_dict = vocab()
print(vocab_dict[w1], vocab_dict[w2], vocab_dict[w3], vocab_dict[w4])
print(cosine_similarity(weights[vocab_dict[w3]], weights[vocab_dict[w4]]))

