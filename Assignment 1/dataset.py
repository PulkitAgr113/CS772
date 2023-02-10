from nltk.corpus import gutenberg
import requests
from bs4 import BeautifulSoup
import re

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

def changed_words():
    return {"Gujurat": "Gujarat", "Telengana": "Telangana",
            "Maharastra": "Maharashtra", "Slovakian": "Slovaks",
            "Kiev" : "Kyiv", "US": "United States"}

def analogy_dataset():
    k = 20
    augmented_sents = []
    disambiguated = disambiguated_words()
    changed = changed_words()
    for word in training_words():
        title = disambiguated.get(word, word)
        title = changed.get(title, title)
        word = changed.get(word, word)
        sentences = get_sents(word, title)
        augmented_sents.extend(sentences[:k])
        print(f"Extracted {len(sentences)} sentences for word {word}")

    with open("analogy_sentences.txt", "w") as file:
        for sent in augmented_sents:
            file.write(sent + "\n")

analogy_dataset()