from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from utils import Pentagram, get_one_hot_encoding, params
from most_similar import Similar

def run_cbow(epoch, model, vocab_dict, batch_size, train_sents, sample_size=1000):
    train_tuples = Pentagram(train_sents)
    train_dataloader = DataLoader(train_tuples, batch_size=batch_size, shuffle=True)
    error = 0
    for idx,item in enumerate(tqdm(train_dataloader)):
        # if idx >= sample_size:
        #     break
        for in_ind in [0,1,3,4]:
            input_vecs = get_one_hot_encoding(vocab_dict, item[in_ind])
            if in_ind == 0:
                hidden_vecs = model.hidden_layer(input_vecs)
            else:
                hidden_vecs += model.hidden_layer(input_vecs)

        hidden_vecs /= 4
        pred_vecs = model.output_layer(hidden_vecs)
        output_vecs = get_one_hot_encoding(vocab_dict, item[2])
        w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd = model.get_update_calc(output_vecs, pred_vecs)
        error += model.get_error(output_vecs, pred_vecs)
        model.update_wb(w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd)
        
    print(f"Epoch {epoch} completed\nTraining Error: {error}")
    return model

if __name__ == '__main__':
    epochs, model, batch_size, vocab_dict, train_sents = params()
        
    for epoch in range(epochs):
        model = run_cbow(epoch, model, vocab_dict, batch_size, train_sents)
        sim = Similar(model.fc1.weights)
        print(f"Validation Accuracy: {sim.accuracy()}")
        with open("cbow_model.pkl", "wb") as file:
            pickle.dump(model, file)
                