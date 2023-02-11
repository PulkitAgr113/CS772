import numpy as np
        
class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # self.weights = np.random.rand(input_size,output_size)-0.5
        self.weights = np.zeros((input_size,output_size))
        self.bias = np.random.rand(output_size)-0.5
        self.input=None

    def forward(self, input):
        '''
        Performs a forward pass of a fully connected network
        Args:
          input : training data, numpy array of shape (n_samples , self.input_size)

        Returns:
           numpy array of shape (n_samples , self.output_size)
        '''
        self.input = input
        return input@self.weights+self.bias        

    def backward(self, output_error):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        output_error_new = output_error@(self.weights.T)
        dev_w = (self.input.T)@output_error
        dev_b = np.squeeze(np.sum(output_error,axis=0))
        return dev_w, dev_b, output_error_new       
    
class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.input = None
    
    def forward(self, input):
        self.input = input
        input = input.clip(min=-700,max=700)
        return np.exp(input)/(np.sum(np.exp(input),axis=1).reshape(-1,1)+1e-9)   
    
def cross_entropy(y_true, y_pred):
    loss = -np.mean(np.log(y_pred+1e-9)*y_true)
    return loss

def cross_entropy_and_softmax_prime(y_true, y_pred):
    '''
    Implements derivative of cross entropy function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of cross entropy function
    '''
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

