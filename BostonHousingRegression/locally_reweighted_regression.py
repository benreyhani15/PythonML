from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses 
 
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    A = compute_distance_weight_matrix(test_datum, x_train, tau)
    w = compute_LR_weights(A, x_train, y_train, lam)
    y_hat = np.dot(test_datum.transpose(),w)
    return y_hat

def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    fold_size = int(N/k)
    total_loss = np.zeros(taus.shape)
    for i in range(k):
        test_start_index = i*fold_size
        test_end_index = N if i==(k-1) else (test_start_index + fold_size)
        x_test = x[test_start_index:test_end_index]
        y_test = y[test_start_index:test_end_index]
        x_train = np.delete(x, np.arange(test_start_index, test_end_index),0)
        y_train = np.delete(y, np.arange(test_start_index, test_end_index), 0)
        #print("taking test indices: {}-->{}".format(test_start_index, test_end_index))
        #print("x_test: {}, y_test: {}, x_train: {}, y_rain: {}".format(x_test.shape, y_test.shape, x_train.shape, y_train.shape))
        new_loss = run_on_fold(x_test, y_test, x_train, y_train, taus)
        total_loss = total_loss + new_loss
    losses = (total_loss/k)
    return losses
                 
def compute_LR_weights(A, x_train, y_train, lam):
    a_1 = np.dot(x_train.transpose(),A).dot(x_train)
    a = a_1 + (lam*np.identity(a_1.shape[0]))
    b = np.dot(x_train.transpose(), A).dot(y_train)
    w = np.linalg.solve(a, b)
    return w

def compute_distance_weight_matrix(test_datum, x_train, tau):
    tmp_test = np.reshape(test_datum, (1,-1))
    squared_norms = l2(x_train, tmp_test)
    scaling_factor = -1/(2*(tau**2))
    ln_of_sum_of_exps = logsumexp(scaling_factor*squared_norms)
    exp_vector = scaling_factor*squared_norms
    A_column = np.exp(exp_vector - ln_of_sum_of_exps)
    #print("sum of weights: {}".format(np.sum(A_column)))
    A = np.diag(A_column[:,0])
    return A

def visualize(taus, losses):
    plt.plot(taus, losses)
    plt.tight_layout()
    plt.xlabel("Taus")
    plt.ylabel("Average Losses")
    plt.title("Average Losses vs. Taus")
    print("min loss = {}".format(losses.min()))
    plt.show()

if __name__ == "__main__":
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    visualize(taus, losses)
