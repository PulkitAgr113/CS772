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
    
    def most_similar(self, ind1, ind2, ind3,ind4):
        vec = self.weight_embeds[ind3]-self.weight_embeds[ind1]+self.weight_embeds[ind2]
        similarity_vecs = cosine_similarity(vec, self.weight_embeds)        
        ind = np.argsort(-similarity_vecs)
        # distance_vecs = euclidean_dist(vec, self.weight_embeds)
        # ind = np.argsort(distance_vecs)
        # print(np.where(ind==ind4)[0])
        # print(self.vocab_dict_inv[ind[0]],self.vocab_dict_inv[ind[1]],self.vocab_dict_inv[ind[2]],self.vocab_dict_inv[ind1],self.vocab_dict_inv[ind2],self.vocab_dict_inv[ind3],self.vocab_dict_inv[ind4])
        # print()
        # print(self.vocab_dict_inv[ind4])
        # print()
        if(ind[0]!=ind1 and ind[0]!=ind2 and ind[0]!=ind3):
            # print("case 1")
            # print(ind[0],ind4)
            # print(self.vocab_dict_inv[ind[0]],self.vocab_dict_inv[ind1],self.vocab_dict_inv[ind2],self.vocab_dict_inv[ind3],self.vocab_dict_inv[ind4])
            print(self.vocab_dict_inv[ind[0]])
            return ind[0]
        elif(ind[1]!=ind1 and ind[1]!=ind2 and ind[1]!=ind3):
            # print("case 2")
            # print(ind[1],ind4)
            # print(self.vocab_dict_inv[ind[1]],self.vocab_dict_inv[ind1],self.vocab_dict_inv[ind2],self.vocab_dict_inv[ind3],self.vocab_dict_inv[ind4])
            print(self.vocab_dict_inv[ind[1]])
            return  ind[1]
        elif(ind[2]!=ind1 and ind[2]!=ind2 and ind[2]!=ind3):
            # print("case 3")
            # print(ind[2],ind4)
            # print(self.vocab_dict_inv[ind[2]],self.vocab_dict_inv[ind1],self.vocab_dict_inv[ind2],self.vocab_dict_inv[ind3],self.vocab_dict_inv[ind4])
            print(self.vocab_dict_inv[ind[2]])
            return  ind[2]
        else:
            # print("case 4")
            # print(ind[3],ind4)
            # print(self.vocab_dict_inv[ind[3]],self.vocab_dict_inv[ind1],self.vocab_dict_inv[ind2],self.vocab_dict_inv[ind3],self.vocab_dict_inv[ind4])
            print(self.vocab_dict_inv[ind[3]])
            return  ind[3]

    def accuracy(self):
        corr = 0
        tot = 0
        changed = changed_words()
        count=0
        with open("Validation.txt", "r") as file:
            for line in file.readlines():
                words = line.strip().split()
                # print(words)
                ind1 = self.vocab_dict.get(changed.get(words[0], words[0]).lower(),0)
                ind2 = self.vocab_dict.get(changed.get(words[1], words[1]).lower(),0)
                ind3 = self.vocab_dict.get(changed.get(words[2], words[2]).lower(),0)
                ind4 = self.vocab_dict.get(changed.get(words[3], words[3]).lower(),0)
                if(ind4==0 or ind1==0 or ind2==0 or ind3==0):
                    count+=1
                    continue
                pred_ind = self.most_similar(ind1, ind2, ind3,ind4)
                pred_word = self.vocab_dict_inv[pred_ind]
                if pred_word == changed.get(words[3], words[3]).lower():
                    corr += 1
                tot += 1
                # print(pred_word, changed.get(words[3], words[3]))
        print(count)
        return corr / tot
    
if __name__ == '__main__':
    with open("skipgram_model_128.pkl", "rb") as file:
        model = pickle.load(file)
    sim = Similar(model.fc1.weights)
    print(sim.accuracy())
                

