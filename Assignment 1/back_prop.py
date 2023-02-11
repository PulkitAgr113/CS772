import numpy as np

        
class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size,output_size)-0.5
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

    def backward(self, output_error, learning_rate):
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
        dev_b = np.squeeze(output_error)
        return dev_w, dev_b, output_error_new
        
        
class ActivationLayer:
    def __init__(self, activation, activation_prime):
        '''
          Args:
          activation : Name of the activation function (sigmoid,tanh or relu)
          activation_prime: Name of the corresponding function to be used during backpropagation (sigmoid_prime,tanh_prime or relu_prime)
        '''
        self.activation = activation
        self.activation_prime = activation_prime
        self.input=None
    
    def forward(self, input):
        self.input = input
        if self.activation==sigmoid:
            return sigmoid(self.input)
        elif self.activation==relu:
            return relu(self.input)
        elif self.activation==tanh:
            return tanh(self.input)
        

    def backward(self, output_error, learning_rate):
        if self.activation_prime==sigmoid_prime:
            return sigmoid_prime(self.input)*output_error
        elif self.activation_prime==relu_prime:
            return relu_prime(self.input)*output_error
        elif self.activation_prime==tanh_prime:
            return tanh_prime(self.input)*output_error
        
    
class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.input = None
    
    def forward(self, input):
        self.input = input
        input = input.clip(min=-700,max=700)
        return np.exp(input)/(np.exp(input).sum())
        
    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a Softmax layer
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        S=np.exp(self.input)/np.exp(self.input).sum()
        I_S=np.diag(np.squeeze(S))
        S_dev=I_S-1*S.T@S
        output_error = output_error@S_dev
        return output_error        
        
def sigmoid(x):
    x= x.clip(min=-700,max=700)
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-tanh(x)**2

def relu(x):
    x[x<0] = 0
    return x

def relu_prime(x):
    x[x>=0]=1
    x[x<0]=0
    return x
    
def cross_entropy(y_true, y_pred):
    loss = (np.log(y_pred+1e-9).T)[y_true]
    return loss

def cross_entropy_prime(y_true, y_pred):
    '''
    Implements derivative of cross entropy function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of cross entropy function
    '''
    y_t=np.zeros(y_pred.shape)
    y_t[0,y_true]=((-1/y_pred).T)[y_true]
    np.reshape(y_t,(1,)+y_t.shape)
    return y_t

# def fit(X_train, Y_train, dataset_name):

#     '''
#     Args:
#         X_train -- np array of share (num_train, vocab_size) 
#         Y_train -- np array of share (num_train, vocab_size)     
#     '''


#     vocab_size = None
#     embedding_dim = None    

#     network = [
#         FCLayer(vocab_size, embedding_dim),
#         ActivationLayer(sigmoid, sigmoid_prime),
#         FCLayer(embedding_dim, vocab_size),
#         SoftmaxLayer(vocab_size)
#     ] 

#     epochs = 40
#     learning_rate = 0.01

#     for epoch in range(epochs):
#         error = 0
#         for x, y_true in zip(X_train, Y_train):
#             output = np.reshape(x,(1,)+x.shape)
#             for layer in network:
#                 output = layer.forward(output)
            
#             error += cross_entropy(y_true, output)
#             output_error = cross_entropy_prime(y_true, output)
#             for layer in reversed(network):
#                 output_error = layer.backward(output_error, learning_rate)
        
#         error /= len(X_train)
#         print('%d/%d, error=%f' % (epoch + 1, epochs, error))

#     network_dict = {}

#     network_dict['network'] = network
#     with open(f'model.pkl', 'wb') as files:
#         pkl.dump(network_dict, files)
    
    
                
        
class Model:
    def __init__(self, vocab_size, embedding_dim, lr):
        self.lr=lr
        self.fc1=FCLayer(vocab_size, embedding_dim)
        self.ac1=ActivationLayer(sigmoid, sigmoid_prime)
        self.fc2=FCLayer(embedding_dim, vocab_size)
        self.sm=SoftmaxLayer(vocab_size)

    def forward(self,input):
        input = np.reshape(input,(1,)+input.shape)
        out=self.fc1.forward(input)
        out=self.ac1.forward(out)
        out=self.fc2.forward(out)
        out=self.sm.forward(out)
        return out

    def get_error(self, y_true, y_pred):
        return cross_entropy(y_true,y_pred)

    def get_update_calc(self, y_true, y_pred):
        output_error=cross_entropy_prime(y_true, y_pred)
        output_error=self.sm.backward(output_error, self.lr)
        dev_w_fc2, dev_b_fc2, output_error=self.fc2.backward(output_error, self.lr)
        output_error=self.ac1.backward(output_error, self.lr)
        dev_w_fc1, dev_b_fc1, output_error=self.fc1.backward(output_error, self.lr)
        return dev_w_fc1, dev_b_fc1, dev_w_fc2, dev_b_fc2
    
    def update_wb(self, dev_w_fc1, dev_b_fc1, dev_w_fc2, dev_b_fc2):
        self.fc1.weights-=self.lr*(dev_w_fc1)
        self.fc1.bias-=self.lr*(dev_b_fc1)
        self.fc2.weights-=self.lr*(dev_w_fc2)
        self.fc2.bias-=self.lr*(dev_b_fc2)

