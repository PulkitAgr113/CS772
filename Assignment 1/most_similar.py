import numpy as np

def cosine_similarity(vec, embeds):
    return (embeds@vec) / (np.linalg.norm(vec) * np.linalg.norm(embeds,axis=1))

class most_similar:
    def __init__(self,weights):
        self.weight_embeds=weights
    
    def most_similar(self,ind1,ind2,ind3):
        vec=self.weight_embeds[ind3]-self.weight_embeds[ind1]+self.weight_embeds[ind2]
        similarity_vecs = cosine_similarity(vec,self.weight_embeds)        
        return np.argmax(similarity_vecs)

