import numpy as np
from matplotlib import pyplot as plt
import math
import pickle as pkl

std_g=None
mean_g=None

# def preprocessing(X):
#     """
#     Args:
#     X : numpy array of shape (n_samples, n_features)
    
#     Returns:
#     X_out: numpy array of shape (n_samples, n_features) after normalization
#     """
#     mean=np.mean(X,axis=0)
#     std=np.std(X,axis=0)
#     global std_g
#     std_g = std
#     global mean_g
#     mean_g = mean
#     X_out=(X-mean)/(std+1e-9)
#     min=np.min(X_out,axis=0)
#     max=np.max(X_out,axis=0)
#     X_out=(X_out-min)/(max-min+1e-9)

#     assert X_out.shape == X.shape

#     return X_out

# def split_data(X, Y, train_ratio=0.8):
#     '''
#     Split data into train and validation sets
#     The first floor(train_ratio*n_sample) samples form the train set
#     and the remaining the validation set

#     Args:
#     X - numpy array of shape (n_samples, n_features)
#     Y - numpy array of shape (n_samples, 1)
#     train_ratio - fraction of samples to be used as training data

#     Returns:
#     X_train, Y_train, X_val, Y_val
#     '''
#     # Try Normalization and scaling and store it in X_transformed

#     X_transformed = preprocessing(X)

#     assert X_transformed.shape == X.shape

#     num_samples = len(X)
#     indices = np.arange(num_samples)
#     num_train_samples = math.floor(num_samples * train_ratio)
#     train_indices = np.random.choice(indices, num_train_samples, replace=False)
#     val_indices = list(set(indices) - set(train_indices))
#     X_train, Y_train, X_val, Y_val = X_transformed[train_indices], Y[train_indices], X_transformed[val_indices], Y[val_indices]
  
#     return X_train, Y_train, X_val, Y_val

# class FlattenLayer:
#     '''
#     This class converts a multi-dimensional into 1-d vector
#     '''
#     def __init__(self, input_shape):
#         '''
#         Args:
#          input_shape : Original shape, tuple of ints
#         '''
#         self.input_shape = input_shape

#     def forward(self, input):
#         '''
#         Converts a multi-dimensional into 1-d vector
#         Args:
#           input : training data, numpy array of shape (n_samples , self.input_shape)

#         Returns:
#           input: training data, numpy array of shape (n_samples , -1)
#         '''
#         ## TODO

#         #Modify the return statement to return flattened input
#         return np.reshape(input,(input.shape[0],-1))
#         ## END TODO
        
    
#     def backward(self, output_error, learning_rate):
#         '''
#         Converts back the passed array to original dimention 
#         Args:
#         output_error :  numpy array 
#         learning_rate: float

#         Returns:
#         output_error: A reshaped numpy array to allow backward pass
#         '''
#         ## TODO

#         #Modify the return statement to return reshaped array
#         return np.reshape(output_error,(output_error.shape[0],)+self.input_shape)
#         ## END TODO
        
        
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
        self.weights -= learning_rate*dev_w
        self.bias -= learning_rate*dev_b
        return output_error_new
        
        
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
    loss = np.sum(-np.choose(y_true, np.log(y_pred+1e-9).T))
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
    y_t[0,y_true]=np.choose(y_true, (-1/y_pred).T)
    np.reshape(y_t,(1,)+y_t.shape)
    return y_t

def mss(y_true, y_pred):
    loss = np.sum((y_true-y_pred)**2)
    return loss

def mss_prime(y_true, y_pred):
    return 2*(y_pred-y_true)


def fit(X_train, Y_train, dataset_name):

    '''
    Args:
        X_train -- np array of share (num_train, vocab_size) 
        Y_train -- np array of share (num_train, vocab_size)     
    '''


    vocab_size = None
    embedding_dim = None    

    network = [
        FCLayer(vocab_size, embedding_dim),
        ActivationLayer(sigmoid, sigmoid_prime),
        FCLayer(embedding_dim, vocab_size),
        SoftmaxLayer(vocab_size)
    ] 

    epochs = 40
    learning_rate = 0.01

    for epoch in range(epochs):
        error = 0
        for x, y_true in zip(X_train, Y_train):
            output = np.reshape(x,(1,)+x.shape)
            for layer in network:
                output = layer.forward(output)
            
            error += cross_entropy(y_true, output)
            output_error = cross_entropy_prime(y_true, output)
            for layer in reversed(network):
                output_error = layer.backward(output_error, learning_rate)
        
        error /= len(X_train)
        print('%d/%d, error=%f' % (epoch + 1, epochs, error))

    network_dict = {}

    network_dict['network'] = network
    with open(f'model.pkl', 'wb') as files:
        pkl.dump(network_dict, files)
    
# def predict(X_test, dataset_name):
#     """

#     X_test -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.

#     This is the function that we will call from the auto grader. 

#     This function should only perform inference, please donot train your models here.

#     Steps to be done here:
#     1. Load your trained model/weights from ./models/{dataset_name}_model.pkl
#     2. Ensure that you read model/weights using only the libraries we have given above.
#     3. In case you have saved weights only in your file, itialize your model with your trained weights.
#     4. Compute the predicted labels and return it

#     Return:
#     Y_test - nparray of shape (num_test,)
#     """
#     Y_test = np.zeros(X_test.shape[0],)

#     ## TODO
#     with open(f'./models/{dataset_name}_model.pkl', 'rb') as files:
#         network_dict = pkl.load(files)

#     network = network_dict['network']
#     mean=network_dict['mean']
#     std=network_dict['std']
#     pred = []

#     # Note: Here I have assumed that predict is taking a raw X_test 
#     # But if it is not taking such an x than please comment the following  line

#     # mean=np.mean(X_test,axis=0)
#     # std=np.std(X_test,axis=0)
#     X_test=(X_test-mean)/(std+1e-9)
#     min=np.min(X_test,axis=0)
#     max=np.max(X_test,axis=0)
#     X_test=(X_test-min)/(max-min+1e-9)

#     # X_test = preprocessing(X_test)

#     for x in X_test:
#         # forward
#         # print(x)
#         output = np.reshape(x,(1,)+x.shape)
#         for layer in network:
#             output = layer.forward(output)
#         pred.append(np.argmax(output,axis=1))    

#     Y_test = np.squeeze(np.array(pred))
    
#     ## END TODO
#     assert Y_test.shape == (X_test.shape[0],) and type(Y_test) == type(X_test), "Check what you return"
#     return Y_test
    
if __name__ == "__main__":    
    np.random.seed(0)
    
    # X_train, Y_train= 

    # write_similar to following
    # preprocessed_X = preprocessing(train_mnist[0])
    # fit(preprocessed_X,train[1],dataset)
