import numpy as np 
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from copy import copy

np.random.seed(1847)

class BatchSampler(object):
    '''
    A simple wrapper to randomly sample batches without replacement.

    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
        vel - momentum velocity
        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, param_dimension, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = np.zeros((param_dimension, 1))

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        vel_prev = copy(self.vel)
        term1 = -(self.lr * grad)
        term2 = (self.beta * vel_prev)
        new_vel = term1.reshape(term1.shape[0], 1) + term2
        new_params = params.reshape(params.shape[0],1) + new_vel
        self.vel = copy(new_vel) 
        return new_params

class SVM(object):
    '''
    A Support Vector Machine
    '''
    # Functions assume that X's first column is for bias and is equal to 1!!!!

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Computes the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        t = np.dot(X, self.w)
        term1 = np.multiply(y, t)
        return np.maximum(np.zeros(term1.shape), (1-term1))    
    
    def get_hinge_grads(self, X, y):
        # Returns mxn vector
        y = y.reshape(y.shape[0],1)
        predictions = np.dot(X, self.w)
        predictions = predictions.reshape(predictions.shape[0],1)
        check_vec = np.multiply(y, predictions)
        mask_vec = (check_vec<1).astype(int)
        value = -np.multiply(y, X)
        return np.multiply(mask_vec, value).T

    def grad(self, X, y):
        '''
        Computes the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        w_regularized = copy(self.w)
        w_regularized[0] = 0
        w = self.w
        N = X.shape[0]
        hinge_grads = self.get_hinge_grads(X,y)
        summed = np.sum(hinge_grads, axis=1)
        term2 = (self.c/N)*summed
        grad = w_regularized.reshape(w_regularized.shape[0],1) + (term2.reshape(term2.shape[0],1))
        return grad

    def classify(self, X):
        '''
        Classifies new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        predictions = np.sign(np.dot(X, self.w))
        return predictions.reshape(predictions.shape[0], 1)
  
    def L(self, X, y, w1):
        w_reg = copy(w1)
        w_reg[0] = 0
        term1 = (1/2)*(np.dot(w_reg, w_reg.T))
        hinge_vec = np.maximum(0,(np.multiply((1-y), np.dot(X, w1))))
        term2 = (self.c/X.shape[0])*(np.sum(hinge_vec))
        return term1 + term2

    def check_grad(self, X, y):
        epsilon = 1.49e-08
        grad = np.zeros(self.w.shape)
        for i in range(grad.shape[0]):
            w_delta = np.zeros(self.w.shape)
            w_delta[i] += epsilon
            diff = self.L(X,y, self.w+w_delta) - self.L(X,y, self.w-w_delta)
            grad[i] = diff/(2*epsilon)
        return grad

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x
    
    w = np.zeros((1,1)) 
    w[0] = w_init
    w_history = np.zeros((steps+1, 1))
    f_history = np.zeros((steps+1, 1))
    w_history[0] = w_init 
    f_history[0] = func(w_init)

    for i in np.arange(1, steps+1):
        grad = func_grad(w)    
        w = optimizer.update_params(w, grad)
        w_history[i] = copy(w)
        f_history[i] = func(w)
    return w_history, f_history



def visualize_plot(y_label, w1, w2, x):
    l1 =  plt.plot(x, w1, 'r', label='Beta = 0')
    l2 = plt.plot(x, w2, 'b', label ='Beta = 0.9')
    plt.xlabel('Time Step')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def test_sgd():
    gd_optimizer = GDOptimizer(1, 1, 0)
    w_test_1, f_test_1 = optimize_test_function(gd_optimizer)
    gd_optimizer = GDOptimizer(1, 1, 0.9)
    w_test_2, f_test_2 = optimize_test_function(gd_optimizer)
    visualize_plot('w',w_test_1, w_test_2, np.arange(0, 201)) 
    visualize_plot('f(w)', f_test_1, f_test_2, np.arange(0,201))

def add_ones_column(X):
    x = np.ones((X.shape[0], X.shape[1]+1))
    x[:,1:] = X
    return x

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    svm = SVM(penalty, train_data.shape[1])
    w = copy(svm.w)
    bs = BatchSampler(train_data, train_targets, batchsize)
    for i in np.arange(iters):
        X, y = bs.get_batch()
        grad = copy(svm.grad(X,y))
        w = copy(optimizer.update_params(w, grad))
        svm.w = copy(w)
    return svm

def format_data(X_train, X_test, y_train, y_test):
    return add_ones_column(X_train), add_ones_column(X_test), y_train.reshape(y_train.shape[0],1), y_test.reshape(y_test.shape[0],1) 

def plot_as_image(w):
    w = w.reshape(28,28)
    plt.imshow(w, cmap='gray')
    plt.show()
    plt.imshow(w)
    plt.show() 
if __name__ == '__main__':
    test_sgd()
    train_data, train_targets, test_data, test_targets = load_data()
    train_data, test_data, train_targets, test_targets = format_data(train_data, test_data, train_targets, test_targets)

    svm1 = optimize_svm(train_data, train_targets, 1, GDOptimizer(0.05, train_data.shape[1], 0), 100, 500)
    svm2 = optimize_svm(train_data, train_targets, 1, GDOptimizer(0.05, train_data.shape[1], 0.1), 100, 500)
    test_predictions1 = svm1.classify(test_data)
    test_predictions2 = svm2.classify(test_data)
    train_predictions1 = svm1.classify(train_data)
    train_predictions2 = svm2.classify(train_data) 
    hinge_loss_train1 = svm1.hinge_loss(train_data, train_targets)
    hinge_loss_test1 = svm1.hinge_loss(test_data, test_targets)
    hinge_loss_train2 = svm2.hinge_loss(train_data, train_targets)
    hinge_loss_test2 = svm2.hinge_loss(test_data, test_targets)
    
    print("Statistics using SVM 1: Train Accuracy: {}, Test Accuracy: {}, Total Training Hinge Loss: {}, Total Test Hinge Loss: {}, Avg Training Loss: {}, Avg Test Loss: {}".format((train_predictions1 == train_targets).mean(),(test_predictions1 == test_targets).mean(), hinge_loss_train1.sum(), hinge_loss_test1.sum(), hinge_loss_train1.mean(), hinge_loss_test1.mean())) 
    print("Statistics using SVM s: Train Accuracy: {}, Test Accuracy: {}, Total Training Hinge Loss: {}, Total Test Hinge Loss: {}, Avg Training Loss: {}, Avg Test Loss: {}".format((train_predictions2 == train_targets).mean(),(test_predictions2 == test_targets).mean(), hinge_loss_train2.sum(), hinge_loss_test2.sum(), hinge_loss_train2.mean(), hinge_loss_test2.mean())) 
    plot_as_image(copy(svm1.w[1:]))
    plot_as_image(copy(svm2.w[1:]))
