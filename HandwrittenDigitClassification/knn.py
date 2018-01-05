import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Computes L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Queries a single test point using the k-NN algorithm

        Returns the digit label provided by the algorithm
        '''

        distances = self.l2_distance(test_point)
        idx = np.argpartition(distances, k)[:k]
        labels = self.train_labels[idx]
        labels = labels.astype(int)
        bincount = np.bincount(labels)
       
        # Check if there is a tie, if there is then break it by decrementing k until majority vote reached
        max_count = np.amax(bincount)
        digits = np.where(bincount == max_count)[0]
        if k>1 and len(digits) > 1:
            # Tie has occured, decrement k value and re-run
            digit = self.query_knn(test_point, k-1) 
        else: 
            # No tie          
            digit = bincount.argmax()

        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Performs 10-fold cross validation to find the best value for k

    '''

    kf = KFold(n_splits=10)
    accuracy_total = np.zeros(k_range.shape)
    for train_index, cv_index in kf.split(train_data):
        X_train, X_cv = train_data[train_index], train_data[cv_index]
        y_train, y_cv = train_labels[train_index], train_labels[cv_index] 
        # Train knn:
        knn = KNearestNeighbor(X_train, y_train) 
        for index, k in enumerate(k_range):
            accuracy_total[index] += classification_accuracy(knn, k, X_cv, y_cv)
    avg_accuracy = accuracy_total/(kf.get_n_splits(train_data))
    k = k_range[np.argmax(avg_accuracy)]       
    print("Highest accuracy = {}, for k = {}".format(np.max(avg_accuracy), k))
    return k

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluates the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    correct_predictions = 0
    eval_size = eval_labels.shape[0]
    for i in np.arange(eval_size):
        predicted_label = knn.query_knn(eval_data[i], k)
        if predicted_label == eval_labels[i]:
            correct_predictions +=1 
    accuracy = correct_predictions/eval_size
    return accuracy

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Classification accracy for k = 1 on the train and test data
    train_1 = classification_accuracy(knn, 1, train_data, train_labels)
    test_1 = classification_accuracy(knn, 1, test_data, test_labels)

    # Classification accuracy for k = 15 on the train and test data
    train_15 = classification_accuracy(knn, 15, train_data, train_labels)
    test_15 = classification_accuracy(knn, 15, test_data, test_labels)

    print("Classification accuracies for k = 1: train data: {}, test data: {}".format(train_1, test_1))
    print("Classification accuracies for k = 15: train data: {}, test data: {}".format(train_15, test_15))

    optimal_k = cross_validation(train_data, train_labels)
    train_optimal = classification_accuracy(knn, optimal_k, train_data, train_labels)
    test_optimal = classification_accuracy(knn, optimal_k, test_data, test_labels)
    print("Classification accuracies for optimal k, k = {}: train data: {}, test data: {}".format(optimal_k, train_optimal, test_optimal))
    
if __name__ == '__main__':
    main()
