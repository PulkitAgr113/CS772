import nltk
import numpy as np

def create_k_folds(tagged_sentence, k=5, seed=0):
    np.random.seed(seed)
    folds = []
    for i in range(k):
        folds.append([])
    # randomly permute the sentences
    np.random.shuffle(tagged_sentence)
    for i, sentence in enumerate(tagged_sentence):
        folds[i % k].append(sentence)
    return folds

def data():
    tagged_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')

    tagset = set([tag for sentence in tagged_sentences for word, tag in sentence])
    tagset = sorted(list(tagset))
    tagidxdict = {tag: idx for idx, tag in enumerate(tagset)}

    sentences = []
    correct_tags = []
    for sentence in tagged_sentences:
        unzipped_sentence = list(zip(*sentence))
        sentences.append(list(map(lambda x: x.lower(), unzipped_sentence[0])))
        correct_tags.append(list(unzipped_sentence[1]))

    folded_sentences = create_k_folds(sentences)
    folded_correct_tags = create_k_folds(correct_tags)
    print("Folds created")

    return tagidxdict, folded_sentences, folded_correct_tags
