from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    #summarize(boston)
    return X,y,features

def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
       plt.subplot(3, 5, i + 1)
       plt.scatter(X[:, i], y,marker ='x', s=5)
       plt.xlabel(features[i])
       plt.ylabel('MEDV')
       plt.title("Feature #{} Against Target".format(i+1))
    
    plt.tight_layout()
    plt.show()

def summarize(dataset):
    # Print basic dataset stats summary to terminal
    print("X dimensions: {}".format(dataset.data.shape))
    print("y dimensions: {}".format(dataset.target.shape))
    print("Feature Names: {}".format(dataset.feature_names))
    print("Description: {}".format(dataset.DESCR))

    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data['output'] = dataset.target
    print(data.describe(percentiles=[]))

def split_dataset(tp, X, y):
    dataset_count = X.shape[0]
    test_idx = np.random.choice(dataset_count, int(dataset_count*tp), replace=False)
    X_test = X[test_idx]
    X_train = np.delete(X, test_idx, 0)
    y_test = y[test_idx]
    y_train = np.delete(y, test_idx)
    #print("X_test: {}, X_train: {}, y_test: {}, y_train: {}".format(X_test.shape, X_train.shape, y_test.shape, y_train.shape))
    return X_train, X_test, y_train, y_test

def scale(X):
    # Using the computed means and stds from summarize function
    means = np.array([0, 3.593, 11.363, 11.137, 0.0691,0.554, 6.284, 68.575, 3.795, 9.549, 408.237, 18.455, 356.674, 12.653])
    stds = np.array([1, 8.597, 23.322, 6.86, 0.254, 0.116, 0.703, 28.149, 2.106, 8.707, 168.537, 2.165, 91.295, 7.141])
    return (X-means)/stds

def fit_regression(X,Y):
    # Using np.linalg.solve to solve (X'X)w=X'Y
    X_trans = np.transpose(X)
    a = np.dot(X_trans,X)
    b = np.dot(X_trans,Y) 
    return np.linalg.solve(a,b)

def compute_MSE(w, X, y):
    diff = y - np.dot(X, w)
    se = np.dot(np.transpose(diff), diff)
    sample_count = X.shape[0]
    return se/sample_count

def compute_MAE(w, X, y):
    diff = y - np.dot(X,w)
    sample_count = X.shape[0]
    ae = np.absolute(diff)
    return np.sum(ae)/sample_count

def compute_R2(w, X, y):
    y_mean = np.mean(y)
    residuals = y - np.dot(X,w)
    variations = y - y_mean
    SS_res = np.dot(np.transpose(residuals), residuals)
    SS_tot = np.dot(np.transpose(variations), variations)
    return 1-(SS_res/SS_tot)

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    # Visualize the features
    visualize(X, y, features)

    # Splitting data set into Train and Test
    X_train, X_test, y_train, y_test = split_dataset(0.2, X, y)   
    
    # Insert column of 1's to input matrices
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)

    # Fit regression model
    w = fit_regression(X_train, y_train)
    print("W: {}".format(w))
    
    # Compute fitted values, MSE, etc.
    mse = compute_MSE(w, X_test, y_test)
    print("mse: {}, mae: {}, R2: {}".format(mse, compute_MAE(w, X_test, y_test), compute_R2(w, X_test, y_test)))

    # Scale features and recompute weights to determine most impactful features via weight magnitudes
    w_scaled = fit_regression(scale(X_train), y_train)    
    print("W_scaled: {}".format(w_scaled))

if __name__ == "__main__":
    main()
