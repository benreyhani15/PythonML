import data
import numpy as np
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarizes the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    Returns a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for i in np.arange(10):
        tmp_data = data.get_digits_by_label(train_data, train_labels, i)
        N_kj = np.sum(tmp_data, axis=0).reshape((1, eta.shape[1]))
        N_k = tmp_data.shape[0]
        eta[i] = (N_kj+1)/(N_k+2)
    return eta

def plot_images(class_images):
    '''
    Plots each of the images corresponding to each class side by side in grayscale
    '''
    array = []
    for i in range(10):
        img_i = class_images[i]
        array.append(img_i.reshape((8,8)))
    all_concat = np.concatenate(array, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show() 

def generate_new_data(eta):
    '''
    Samples a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    And then plots these values
    '''
    generated_data = np.random.binomial(1, eta)
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Computes the generative log-likelihood:
        log p(x|y, eta)

    Returns an n x 10 numpy array 
    '''
    return (np.log(eta).dot(bin_digits.T) + np.log(1-eta).dot(1-bin_digits.T)).T    
    
def conditional_likelihood(bin_digits, eta):
    '''
    Computes the conditional likelihood:

        log p(y|x, eta)

    This is a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_LLs = generative_likelihood(bin_digits, eta)
    generative_probs = generative_probabilities(bin_digits, eta)
    Px = np.sum(generative_probs, axis=1)
    log_Px = np.log(Px).reshape((Px.shape[0],1))
    return (gen_LLs - log_Px)

def generative_probabilities(bin_digits, eta):
    # Returns a nx10 matrix with each data point evaluated at the bernoulli probability for each class
    probs = np.zeros((bin_digits.shape[0], eta.shape[0]))
    for i in np.arange(probs.shape[0]):
        for j in np.arange(probs.shape[1]):
            x = bin_digits[i]
            params = eta[j]
            product = np.multiply(np.power(params, x), np.power((1-params), (1-x)))
            probs[i,j] = np.prod(product)
    return probs 

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Computes the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    N = cond_likelihood.shape[0]
    extracted_cond_likelihoods = cond_likelihood[np.arange(N), labels.astype(int)]
    #test_cond_likelihoods(cond_likelihood, extracted_cond_likelihoods)
    return extracted_cond_likelihoods.mean() 

def test_cond_likelihoods(cond_likelihood, extracted_cond_likelihoods):
    probs = np.exp(cond_likelihood)
    correct_probs = np.exp(extracted_cond_likelihoods)
    min = probs.min()
    max = probs.max()
    sum = np.sum(probs, axis=1)
    correct_mean = correct_probs.mean()
    print("mean of data entry probability: {}, correct probability mean: {}, min probability: {}, max probability: {}".format(sum.mean(), correct_mean, min, max))

def classify_data(bin_digits, eta):
    '''
    Classifies new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    return np.argmax(cond_likelihood, axis=1)

def classification_accuracy(bin_digits, labels, eta):
    predictions = classify_data(bin_digits, eta)
    n = bin_digits.shape[0]
    correct_estimates = np.sum(predictions==labels)
    return correct_estimates/n

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)

    # Average Conditional Log Likelihood:
    train_avg_conditional_LL = avg_conditional_likelihood(train_data, train_labels, eta)
    test_avg_conditional_LL = avg_conditional_likelihood(test_data, test_labels, eta)
    print("Avg Conditional Log Likelihood on: Training data: {}, Testing data: {}".format(train_avg_conditional_LL, test_avg_conditional_LL))
    
    # Classification Accuracy:
    train_accuracy = classification_accuracy(train_data, train_labels, eta)
    test_accuracy = classification_accuracy(test_data, test_labels, eta)
    print("Classification Accuracy on: Training data: {}, Testing data: {}".format(train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()
