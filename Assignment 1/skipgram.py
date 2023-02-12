from tqdm import tqdm
import pickle,os,argparse
from torch.utils.data import DataLoader
from utils import Pentagram, get_one_hot_encoding, params
from most_similar import Similar

print(os.getpid())

def run_skipgram(epoch, model, vocab_dict, batch_size, train_sents, sample_size=1000):
    train_tuples = Pentagram(train_sents)
    train_dataloader = DataLoader(train_tuples, batch_size=batch_size, shuffle=True)
    error = 0
    for idx,item in enumerate(tqdm(train_dataloader)):
        # if idx >= sample_size:
        #     break
        input_vecs = get_one_hot_encoding(vocab_dict, item[2])
        pred_vecs = model.output_layer(model.hidden_layer(input_vecs))

        for out_ind in [0,1,3,4]:
            output_vecs = get_one_hot_encoding(vocab_dict, item[out_ind])
            dev_w_fc1, dev_b_fc1, dev_w_fc2, dev_b_fc2 = model.get_update_calc(output_vecs, pred_vecs)
            error += model.get_error(output_vecs, pred_vecs)
            if out_ind == 0:
                w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd = dev_w_fc1, dev_b_fc1, dev_w_fc2, dev_b_fc2
            else:
                w_fc1_upd += dev_w_fc1
                b_fc1_upd += dev_b_fc1
                w_fc2_upd += dev_w_fc2
                b_fc2_upd += dev_b_fc2

        model.update_wb(w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd)
        
    error /= 4
    print(f"Epoch {epoch} completed. Error: {error}")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs',type=int)
    parser.add_argument('--epoch',type=int)
    args = parser.parse_args()
    epochs, model, batch_size, vocab_dict, train_sents = params()
    batch_size = args.bs    
    epochs = args.epoch
    for epoch in range(epochs):
        if(epoch==0 and os.path.exists(f'skipgram_model_{batch_size}.pkl')):
            with open(f"skipgram_model_{batch_size}.pkl", "rb") as file:
                model=pickle.load(file)
        model = run_skipgram(epoch, model, vocab_dict, batch_size, train_sents)
            
        sim = Similar(model.fc1.weights)
        print(f"Validation Accuracy: {sim.accuracy()}")
        with open(f"skipgram_model_{batch_size}.pkl", "wb") as file:
            pickle.dump(model, file)
        
                