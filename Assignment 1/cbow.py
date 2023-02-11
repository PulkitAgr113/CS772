import numpy as np
import json
from back_prop import Model
from tqdm import tqdm

def vocab():
    with open("vocab.json", 'r') as file:
        vocab_dict = json.load(file)
    return vocab_dict

def run(epoch, model, vocab_dict, train_sents):
    vocab_size = len(vocab_dict)
    tot_err = 0
    sent_ind = 0
    for line in tqdm(train_sents):
        sent_err = 0
        words = line.strip().split()
        input_vec = np.zeros(vocab_size)
        for ind in range(2, len(words)-2):
            for in_ind in range(ind-2, ind+3):
                if in_ind == ind:
                    continue
                input_word = words[in_ind]
                input_ind = int(vocab_dict[input_word])
                input_vec[input_ind] = 1
                if in_ind == ind-2:
                    hidden_vec = model.hidden_layer(input_vec)
                else:
                    hidden_vec += model.hidden_layer(input_vec)
                input_vec[input_ind] = 0

            hidden_vec /= 4
            pred_vec = model.output_layer(hidden_vec)
            err = 0
            output_word = words[ind]
            output_ind = int(vocab_dict[output_word])
            w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd = model.get_update_calc(output_ind, pred_vec)
            err += model.get_error(output_ind, pred_vec)
            sent_err += err
            model.update_wb(w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd)

        tot_err += sent_err
        sent_ind += 1
        if sent_ind == 1000:
            print(f"Error for last 1000 sentences: {tot_err / sent_ind}")
            tot_err = 0
            sent_ind = 0
        
    print(f"Epoch {epoch} completed.")
    return model

if __name__ == '__main__':
    vocab_dict = vocab()
    embedding_dim = 100
    lr = 0.1
    epochs = 1
    model = Model(len(vocab_dict), embedding_dim, lr)

    with open("training_sents.txt", 'r') as file:
        train_sents = file.readlines()
        
    for epoch in range(epochs):
        model = run(epoch, model, vocab_dict, train_sents)
                