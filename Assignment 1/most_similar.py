import numpy as np
import pickle
from tqdm import tqdm
from utils import changed_words, vocab, vocab_inv

def cosine_similarity(vec, embeds):
    return (embeds@vec) / (np.linalg.norm(vec) * np.linalg.norm(embeds,axis=1) + 1e-9)

def euclidean_dist(vec, embeds):
    return np.linalg.norm(vec-embeds, axis=1)

class Similar:
    def __init__(self,weights):
        self.weight_embeds=weights.detach().cpu().numpy()
        self.vocab_dict = vocab()
        self.vocab_dict_inv = vocab_inv()
    
    def most_similar(self, ind1, ind2, ind3):
        vec = self.weight_embeds[ind3]-self.weight_embeds[ind1]+self.weight_embeds[ind2]
        similarity_vecs = cosine_similarity(vec, self.weight_embeds)        
        ind = np.argsort(similarity_vecs)
        # distance_vecs = euclidean_dist(vec, self.weight_embeds)
        # ind = np.argsort(-distance_vecs)
        if(ind[-1]!=ind1 and ind[-1]!=ind2 and ind[-1]!=ind3):
            return ind[-1]
        elif(ind[-2]!=ind1 and ind[-2]!=ind2 and ind[-2]!=ind3):
            return  ind[-2]
        elif(ind[-3]!=ind1 and ind[-3]!=ind2 and ind[-3]!=ind3):
            return  ind[-3]
        else:
            return  ind[-4]

    def accuracy(self):
        corr = 0
        tot = 0
        changed = changed_words()
        with open("Validation.txt", "r") as file:
            for line in file.readlines():
                words = line.strip().split()
                ind1 = self.vocab_dict.get(changed.get(words[0], words[0]),0)
                ind2 = self.vocab_dict.get(changed.get(words[1], words[1]),0)
                ind3 = self.vocab_dict.get(changed.get(words[2], words[2]),0)
                # ind4 = self.vocab_dict.get(changed.get(words[3], words[3]),0)
                pred_ind = self.most_similar(ind1, ind2, ind3)
                pred_word = self.vocab_dict_inv[pred_ind]
                if pred_word == changed.get(words[3], words[3]):
                    corr += 1
                tot += 1
                print(pred_word, changed.get(words[3], words[3]))

        return corr / tot
    
if __name__ == '__main__':
    with open("skipgram_model.pkl", "rb") as file:
        model = pickle.load(file)
    sim = Similar(model.fc1.weights)
    print(sim.accuracy())
                

