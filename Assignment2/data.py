import nltk
from models import *
from collections import Counter, OrderedDict

nltk.download('universal_tagset')
from nltk.corpus import brown
from nltk.data import find

tag_sents = brown.tagged_sents(tagset='universal')

brown_sents = []
brown_tags = []
brown_words = []
for tag_sent in tag_sents:
    brown_sents.append([])
    brown_tags.append([])
    for tuple in tag_sent:
        brown_words.append(tuple[0].lower())
        brown_sents[-1].append(tuple[0].lower())
        brown_tags[-1].append(tuple[1])


brown_counter = Counter(brown_words)
sorted_by_freq_tuples = sorted(brown_counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
in_vocab = {}
vc = 0
for odi in ordered_dict:
    in_vocab[odi] = vc
    vc+=1


out_vocab = {'DET': 0,
'NOUN': 1,
'ADJ': 2,
'VERB': 3,
'ADP': 4,
'ADV': 5,
'CONJ': 6,
'PRT': 7,
'PRON': 8,
'NUM': 9,
'X': 10,
'.': 11}

dataset_inputs = []
dataset_outputs = []

for i in range(len(brown_sents)):
    dataset_inputs.append([])
    dataset_outputs.append([])
    for j in range(len(brown_sents[i])):
        dataset_inputs[-1].append(in_vocab[brown_sents[i][j]])
        print(brown_tags[i][j])
        print(out_vocab[brown_tags[i][j]])
        dataset_outputs[-1].append(out_vocab[brown_tags[i][j]])

    break

INPUT_DIM = len(in_vocab)
OUTPUT_DIM = len(out_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, torch.device('cpu')).to(torch.device('cpu'))


src = dataset_inputs[0]
trg = dataset_outputs[0]

src = torch.tensor(src)
trg = torch.tensor(trg)
src = src.reshape((-1,1))
trg = trg.reshape((-1,1))


output = model(src, trg)
#trg = [trg len, batch size]
#output = [trg len, batch size, output dim]

output_dim = output.shape[-1]

output = output[1:].view(-1, output_dim)
trg = trg[1:].view(-1)

#trg = [(trg len - 1) * batch size]
#output = [(trg len - 1) * batch size, output dim]

print(output)
print(brown_tags[0])
print(trg)