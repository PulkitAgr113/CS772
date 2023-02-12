import numpy as np
from tqdm import tqdm
from utils import changed_words, vocab

def cosine_similarity(vec, embeds):
    return (embeds@vec) / (np.linalg.norm(vec) * np.linalg.norm(embeds,axis=1) + 1e-9)

class Similar:
    def __init__(self,weights):
        self.weight_embeds=weights
        self.vocab_dict = vocab()
    
    def most_similar(self, ind1, ind2, ind3):
        vec = self.weight_embeds[ind3]-self.weight_embeds[ind1]+self.weight_embeds[ind2]
        similarity_vecs = cosine_similarity(vec, self.weight_embeds)        
        ind = np.argpartition(similarity_vecs, -5)[-5:]
        # ind=np.argmax(similarity_vecs)     
        if ind[0]!=ind1:
            return ind[0]
        else:
            return ind[1]
    
    def accuracy(self):
        corr = 0
        tot = 0
        changed = changed_words()
        with open("Validation.txt", "r") as file:
            for line in tqdm(file.readlines()):
                words = line.strip().split()
                ind1 = self.vocab_dict.get(changed.get(words[0], words[0]),0)
                ind2 = self.vocab_dict.get(changed.get(words[0], words[0]),0)
                ind3 = self.vocab_dict.get(changed.get(words[0], words[0]),0)
                pred_ind = self.most_similar(ind1, ind2, ind3)
                if [word for word in self.vocab_dict if self.vocab_dict[word]==pred_ind] == changed.get(words[3], words[3]):
                    corr += 1
                tot += 1

        return corr / tot

                

