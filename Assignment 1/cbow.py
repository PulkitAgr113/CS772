from tqdm import tqdm
import pickle, os, argparse, torch
from torch.utils.data import DataLoader
from utils import nGram, get_one_hot_encoding, params
from most_similar import Similar

print(os.getpid())

def run_cbow(epoch, model, vocab_dict, batch_size, train_sents, n=2, sample_size=5000):
    train_tuples = nGram(train_sents, n)
    train_dataloader = DataLoader(train_tuples, batch_size=batch_size, shuffle=True)
    error = 0
    for idx,item in enumerate(tqdm(train_dataloader)):
        # if idx >= sample_size:
        #     break
        for in_ind in range(2*n+1):
            if in_ind == n:
                continue
            input_vecs = get_one_hot_encoding(vocab_dict, item[in_ind])
            if in_ind == 0:
                avg_vecs = input_vecs
            else:
                avg_vecs += input_vecs

        avg_vecs /= (2*n)
        pred_vecs = model.output_layer(model.hidden_layer(avg_vecs))
        output_vecs = get_one_hot_encoding(vocab_dict, item[n])
        w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd = model.get_update_calc(output_vecs, pred_vecs)
        model.update_wb(w_fc1_upd, b_fc1_upd, w_fc2_upd, b_fc2_upd)

        error += model.get_error(output_vecs, pred_vecs)
        
    print(f"Epoch {epoch} completed\nTraining Error: {error}")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs',type=int)
    parser.add_argument('--epoch',type=int)
    args = parser.parse_args()
    epochs, model, batch_size, vocab_dict, train_sents = params("all_training_sents.txt")
    batch_size = args.bs    
    epochs = args.epoch
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        if(epoch==0 and os.path.exists(f'cbow_model_{batch_size}_ed_700.pkl')):
            with open(f"cbow_model_{batch_size}.pkl", "rb") as file:
                model = pickle.load(file)
                model.fc1.weights=model.fc1.weights.cpu().to(device)
                model.fc2.weights=model.fc2.weights.cpu().to(device)
                model.fc1.bias=model.fc1.bias.cpu().to(device)
                model.fc2.bias=model.fc2.bias.cpu().to(device)                
        model = run_cbow(epoch, model, vocab_dict, batch_size, train_sents, 2)
        sim = Similar(model.fc1.weights)
        print(f"Validation Accuracy: {sim.accuracy()}")
        with open(f"cbow_model_{batch_size}_ed_700.pkl", "wb") as file:
            pickle.dump(model, file)
                