import data
import numpy as np
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Computes the mean estimate for each digit class

    Returns a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    number_of_classes = means.shape[0]
    for i in np.arange(number_of_classes):
        tmp_data = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(tmp_data, axis=0)
   
    # Check for correct mean computation 
    visualize_means(means)
    return means

# Used to validate that means were correctly computed
def visualize_means(means):
    mean_array = []
    for i in np.arange(means.shape[0]):
        mean_array.append(means[i].reshape((8,8)))
    all_concat = np.concatenate(mean_array, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def compute_sigma_mles(train_data, train_labels):
    '''
    Computes the covariance estimate for each digit class

    Return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    for i in np.arange(covariances.shape[0]):
        tmp_data = data.get_digits_by_label(train_data, train_labels, i)
        mean = np.mean(tmp_data, axis = 0)
        diff = tmp_data-mean
        number_of_entries = tmp_data.shape[0]
        for j in np.arange(number_of_entries):
            row_data = diff[j].reshape((64,1))
            covariances[i] += row_data.dot(row_data.T)
        covariances[i] = covariances[i]/(number_of_entries)
        # check_cov_calc(covariances[i], tmp_data)
        # Add identity matrix for stability
        covariances[i]+= 0.01*np.identity(covariances[i].shape[0])
    return covariances

# Check to see if cov matrix is implemented correctly 
def check_cov_calc(covariance, data):
    diag = np.all(covariance.diagonal()>=0)
    symmetry = np.allclose(covariance, covariance.T, atol=1e-8)
    print("Diagonal values positive: {}, is matrix symmetrical: {}".format(diag, symmetry))
    np_cov = np.cov(data.T)
    diff = np.abs(covariance-np_cov)
    print("Max between formula and numpy cov: {}".format(np.max(diff)))
    print("Avg diff: {}".format(np.mean(diff)))

def plot_cov_diagonal(covariances):
    # Plots the log-diagonal of each covariance matrix side by side
    cov_image_array = []
    for i in range(covariances.shape[0]):
        cov_diag = np.diag(covariances[i])
        cov_image_array.append(np.log(cov_diag).reshape((8,8)))
    all_concat = np.concatenate(cov_image_array, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def compute_gaussian_probabilities(digits, means, covariances):
    # Returns nx10 matrix with each datapoint evaluated at gaussian of each class
    d = digits.shape[1]
    mahalanobis_distance_matrix = compute_mahalanobis_distance(digits, means, covariances)
    determinants = np.linalg.det(covariances).reshape((1,10))
    exponentials = np.exp(mahalanobis_distance_matrix)
    first_term = np.power(2*np.pi, (-d/2))
    second_term = np.power(determinants, (-1/2))
    gaussian_probs = first_term * (np.multiply(second_term, exponentials))
    return gaussian_probs    
     
def compute_mahalanobis_distance(digits, means, covariances):
    # Returns nx10 matrix computing all the distances for each data entry 
    n = digits.shape[0]
    k = covariances.shape[0]
    distances = np.zeros((n,k))
    inverses = np.linalg.inv(covariances)
    for i in np.arange(n):
        for j in np.arange(k):
            diff = digits[i] - means[j]
            distances[i,j] = (-1/2)*np.dot(np.dot(diff.T, inverses[j]),diff)
    return distances
            
def generative_likelihood(digits, means, covariances):
    '''
    Computes the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Returns an n x 10 numpy array 
    '''
    d = digits.shape[1]
    determinants = np.linalg.det(covariances).reshape((1,10))
    mahalanobis_distance_matrix = compute_mahalanobis_distance(digits, means, covariances)
    term_1 = (-d/2)*np.log(2*np.pi)
    term_2 = (-1/2)*np.log(determinants)
    gen_likelihood = term_1 + (term_2 + mahalanobis_distance_matrix)
    return gen_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Computes the conditional likelihood:

        log p(y|x, mu, Sigma)

    Returns a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    N = digits.shape[0]
    probabilities = compute_gaussian_probabilities(digits, means, covariances)
    summed_generative_liklihood = np.sum(probabilities, axis=1)
    gen_likelihood = generative_likelihood(digits, means, covariances)
    logged_sum = np.log(summed_generative_liklihood).reshape((summed_generative_liklihood.shape[0],1))
    cond_likelihoods = gen_likelihood - logged_sum
    return cond_likelihoods

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Computes the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    N = cond_likelihood.shape[0]
    extracted_cond_likelihoods = cond_likelihood[np.arange(N), labels.astype(int)]
    #test_cond_likelihoods(cond_likelihood, extracted_cond_likelihoods)
    return extracted_cond_likelihoods.mean()             

def classify_data(digits, means, covariances):
    '''
    Classifies new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return np.argmax(cond_likelihood, axis=1)

def classification_accuracy(digits, labels, means, covariances):
    predictions = classify_data(digits, means, covariances)
    n = digits.shape[0]
    correct_estimates = np.sum(predictions==labels)
    return correct_estimates/n    

def test_cond_likelihoods(cond_likelihood, extracted_cond_likelihoods):
    probs = np.exp(cond_likelihood)
    correct_probs = np.exp(extracted_cond_likelihoods)
    min = probs.min()
    max = probs.max()
    sum = np.sum(probs, axis=1)
    correct_mean = correct_probs.mean()
    print("mean of data entry probability: {}, correct probability mean: {}, min probability: {}, max probability: {}".format(sum.mean(), correct_mean, min, max))

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)

    # Evaluation
    # Average conditional log liklihood:
    train_avg = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("Avg conditional likelihood for: train_avg: {}, test_avg: {}".format(train_avg, test_avg))

    # Classification accuracy:
    train_acc = classification_accuracy(train_data, train_labels, means, covariances)
    test_acc = classification_accuracy(test_data, test_labels, means, covariances)
    print("Classification accuracy for: Train: {}, Test: {}".format(train_acc, test_acc))

if __name__ == '__main__':
    main()
