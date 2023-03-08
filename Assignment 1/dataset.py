from nltk.corpus import gutenberg
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import json
from utils import changed_words

def get_gutenberg_sents():
    fileids = gutenberg.fileids()
    gutenberg_sents = []
    for file in fileids:
        gutenberg_sents.extend(gutenberg.sents(file))
    return gutenberg_sents

def training_words():
    analogy_words = set()
    with open("Analogy_dataset.txt", "r") as file:
        for line in file:
            for word in line.strip().split():
                analogy_words.add(word)
    return analogy_words

def get_sents(word, title):
    response = requests.get(
        url=f"https://en.wikipedia.org/wiki/{title}",
    )
    soup = BeautifulSoup(response.content, 'html.parser')   
    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.text
    
    text = re.sub('[^a-zA-Z\?\.!\'\",\-()\n\t ]', '', text)
    sentences = re.split('\.|\?|!|\n|\t', text)
    results = []
    for sentence in sentences:
        if word in sentence or word.lower() in sentence:
            results.append(sentence)

    results.sort(key=lambda x: len(re.findall('[a-zA-Z]', x)), reverse=True)
    return results

def disambiguations():
    with open("disambiguation.txt", "w") as file:
        for word in training_words():
            sentences = get_sents(word, word)
            if len(sentences) <= 10:
                print(f"Disambiguation/Change needed for word {word}")
                file.write(word + "\n")

def disambiguated_words():
    disambiguated = {}
    with open("disambiguation.txt", "r") as file:
        for line in file:
            word = line.split("\t")[0]
            dis_word = line.split("\t")[1]
            disambiguated[word] = dis_word.strip()
    return disambiguated

def analogy_dataset():
    augmented_sents = []
    disambiguated = disambiguated_words()
    changed = changed_words()
    new_words = training_words()
    for word in new_words:
        title = disambiguated.get(word, word)
        title = changed.get(title, title)
        word = changed.get(word, word)
        sentences = get_sents(word, title)
        augmented_sents.extend(sentences)
        print(f"Extracted {len(sentences)} sentences for word {word}")

    with open("analogy_sentences.txt", "w") as file:
        for sent in augmented_sents:
            file.write(sent + "\n")

    print(f"Sentence creation from training data completed")

def clean_sents(sents):
    stop_words = set(stopwords.words('english'))
    filtered_sents = []
    for sent in sents:
        filtered_sent = []
        for w in sent:
            if w not in stop_words and re.match('[a-zA-Z]+',w) and len(w)>1:
                if len(wn.synsets(w))>0 or w in training_words() or w in changed_words().values():
                    filtered_sent.append(w.lower())
        if len(filtered_sent)>4:
            filtered_sents.append(filtered_sent)
    return filtered_sents

def get_merged_sents():
    merged_sents=[]
    with open('analogy_sentences.txt') as f:
        analogy_sents = f.readlines()
    a_sents=[]
    for sent in analogy_sents: 
        a_sents.append(re.split(' |\(|\)|-|\'|\"|,', sent.strip())) 

    analogy_sents=a_sents
    cleaned_sents_analogy=clean_sents(analogy_sents)
    merged_sents.extend(cleaned_sents_analogy)
    print("Pre-processing of augmented sentences completed")
    cleaned_sents_gutenberg=clean_sents(get_gutenberg_sents())
    merged_sents.extend(cleaned_sents_gutenberg)
    print("Pre-processing of Gutenberg sentences completed")

    with open("training_sents.txt", "w") as file:
        for sent in merged_sents:
            for word in sent:
                file.write(word + " ")
            file.write("\n")

def get_vocab():
    merged_sents = []
    with open("training_sents.txt", "r") as file:
        for sent in file:
            merged_sents.append(sent.strip().split())

    vocab = set()
    for sent in merged_sents:
        for word in sent:
            vocab.add(word)

    vocab = list(vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    vocab_dict={}
    for i in range(vocab_size):
        vocab_dict[vocab[i]]=i
    
    with open("vocab.json", "w") as file:
        json.dump(vocab_dict, file)

    print("Created vocabulary")                

# analogy_dataset()
get_merged_sents()
get_vocab()