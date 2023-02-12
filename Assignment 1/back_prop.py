import numpy as np
import torch
        
class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # self.weights = np.random.rand(input_size,output_size)-0.5
        # self.bias = np.random.rand(output_size)-0.5
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights = torch.rand(input_size,output_size).to(device)-0.5
        self.bias = torch.rand(output_size).to(device)-0.5
        self.input=None

    def forward(self, input):
        self.input = input
        return input@self.weights+self.bias  

    def backward(self, output_error):
        output_error_new = output_error@(self.weights.T)
        dev_w = (self.input.T)@output_error/self.input.shape[0]
        # dev_b = np.squeeze(np.sum(output_error,axis=0))/self.input.shape[0]
        dev_b = torch.squeeze(torch.sum(output_error,axis=0))/self.input.shape[0]
        return dev_w, dev_b, output_error_new       
    
class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.input = None
    
    def forward(self, input):
        self.input = input
        input = input.clip(min=-350,max=350)
        # return np.exp(input-np.max(input,axis=1).reshape(-1,1))/(np.sum(np.exp(input-np.max(input,axis=1).reshape(-1,1)),axis=1).reshape(-1,1)+1e-9)   
        return torch.exp(input-torch.max(input,axis=1)[0].reshape(-1,1))/(torch.sum(torch.exp(input-torch.max(input,axis=1)[0].reshape(-1,1)),axis=1).reshape(-1,1)+1e-9)   
    
def cross_entropy(y_true, y_pred):
    loss = - torch.sum(torch.log(y_pred+1e-9)*y_true) / len(y_true)
    return loss

def cross_entropy_and_softmax_prime(y_true, y_pred):
    y_t=y_pred-y_true
    return y_t                
        
class Model:
    def __init__(self, vocab_size, embedding_dim, lr):
        self.lr=lr
        self.fc1=FCLayer(vocab_size, embedding_dim)
        self.fc2=FCLayer(embedding_dim, vocab_size)
        self.sm=SoftmaxLayer(vocab_size)

    def hidden_layer(self,input):
        out=self.fc1.forward(input)
        return out
    
    def output_layer(self,input):
        out=self.fc2.forward(input)
        out=self.sm.forward(out)
        return out

    def get_error(self, y_true, y_pred):
        return cross_entropy(y_true,y_pred)

    def get_update_calc(self, y_true, y_pred):
        output_error=cross_entropy_and_softmax_prime(y_true, y_pred)
        dev_w_fc2, dev_b_fc2, output_error=self.fc2.backward(output_error)
        dev_w_fc1, dev_b_fc1, output_error=self.fc1.backward(output_error)
        return dev_w_fc1, dev_b_fc1, dev_w_fc2, dev_b_fc2
    
    def update_wb(self, dev_w_fc1, dev_b_fc1, dev_w_fc2, dev_b_fc2):
        self.fc1.weights-=self.lr*(dev_w_fc1)
        self.fc1.bias-=self.lr*(dev_b_fc1)
        self.fc2.weights-=self.lr*(dev_w_fc2)
        self.fc2.bias-=self.lr*(dev_b_fc2)

