import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
import plotter

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def evaluate_using_linear_svm(train, train_target, test, test_target, c):
    linear_svm = LinearSVC(C=c)
    linear_svm.fit(train, train_target)
    linear_pred_train = linear_svm.predict(train)
    linear_pred_test = linear_svm.predict(test)
    print("Standard Linear SVM: train accuracy = {}, test accuracy = {}\n\n".format((train_target == linear_pred_train).mean(), (test_target == linear_pred_test).mean()))
    return linear_pred_train, linear_pred_test
'''
    lasso_lin_svm = LinearSVC(penalty='l1', loss='squared_hinge',dual=False)
    lasso_lin_svm.fit(train, train_target)
    lasso_pred_train = lasso_lin_svm.predict(train)
    lasso_pred_test = lasso_lin_svm.predict(test)
    print("Lasso Linear SVM: train accuracy = {}, test accuracy = {}".format((train_target == lasso_pred_train).mean(), (test_target == lasso_pred_test).mean()))

    hinge_lin_svm = LinearSVC(penalty='l2', loss='hinge',dual=True)
    hinge_lin_svm.fit(train, train_target)
    hinge_pred_train = hinge_lin_svm.predict(train)
    hinge_pred_test = hinge_lin_svm.predict(test)
    print("Hinge Linear SVM: train accuracy = {}, test accuracy = {}".format((train_target == hinge_pred_train).mean(), (test_target == hinge_pred_test).mean()))
'''

def kfold_cv_lin_svm(k, train, target):
    kf = KFold(n_splits = k, shuffle=True)
    c_array = np.array([0.4, 0.5, 0.6])
    accuracies = np.zeros((k,c_array.shape[0]))
    k_idx = 0
    count = 0
    for train_index, cv_index in kf.split(train):
        print("new fold")
        X_train, X_cv = train[train_index], train[cv_index]
        Y_train, Y_cv = target[train_index], target[cv_index]
        for i, c in enumerate(c_array): 
            print("Computation: {} of {}".format(count, (k*c_array.shape[0])))
            svm = LinearSVC(C=c)
            svm.fit(X_train, Y_train)
            prediction = svm.predict(X_cv)
            accuracies[k_idx,i]=(Y_cv == prediction).mean()
            count += 1
        k_idx+=1
    means = np.mean(accuracies, axis=0)
    print("min: {} for {}, max: {} for {}".format(means.min(), c_array[np.argmin(means)], means.max(), c_array[np.argmax(means)]))
    max_idx = np.argmax(means)
    return c_array[max_idx]

def kfold_cv_lin_svm_lasso(k, train, target):
    kf = KFold(n_splits = k, shuffle=True)
    c_array = np.array([1.2, 1.25, 1.3])
    dimensions = np.zeros((k, c_array.shape[0]))
    accuracies = np.zeros((k,c_array.shape[0]))
    k_idx = 0
    count = 0
    for train_index, cv_index in kf.split(train):
        print("new fold")
    for i, j in enumerate(most_confused):
        X_train, X_cv = train[train_index], train[cv_index]
        Y_train, Y_cv = target[train_index], target[cv_index]
        for i, c in enumerate(c_array):
            print("Computation: {} of {}".format(count, (k*c_array.shape[0])))
            svm = LinearSVC(C=c, penalty="l1", dual=False)
            svm.fit(X_train, Y_train)
            model = SelectFromModel(svm, prefit=True)
            reduced_dimension = model.transform(X_train).shape[1]
            dimensions[k_idx, i] = reduced_dimension
            prediction = svm.predict(X_cv)
            accuracies[k_idx,i]=(Y_cv == prediction).mean()
            count+=1
        k_idx+=1
    means = np.mean(accuracies, axis=0)
    print("min: {} for {}, max: {} for {}".format(means.min(), c_array[np.argmin(means)], means.max(), c_array[np.argmax(means)]))
    max_idx = np.argmax(means)
    return c_array[max_idx]

def evaluate_using_linear_lasso_svm(X_train, Y_train, X_test, Y_test, c):
    svm = LinearSVC(C=c, penalty='l1', dual=False)
    svm.fit(X_train, Y_train)
    train_predictions = svm.predict(X_train)
    test_predictions = svm.predict(X_test)
    print("Lasso Linear SVM: train: {}, test: {}".format((train_predictions == Y_train).mean(), (test_predictions == Y_test).mean()))
    model = SelectFromModel(svm, prefit=True)
    return model.transform(X_train), model.transform(X_test)

def kfold_cv_rbf_svm(k, train, target):
    count = 0
    kf = KFold(n_splits = k, shuffle = True)
    gamma_range = np.array([0.05, 0.1, 0.5])
    C_range = np.array([10])
    accuracies = np.zeros((C_range.shape[0], gamma_range.shape[0]))
    for train_index, cv_index in kf.split(train):
        X_train, X_cv = train[train_index], train[cv_index]
        Y_train, Y_cv = target[train_index], target[cv_index]
        for i, c in enumerate(C_range):
            for j, g in enumerate(gamma_range):
                count+=1
                print("Computation: {} of {}".format(count, (k*C_range.shape[0]*gamma_range.shape[0])))
                rbf_svm = SVC(C=c, gamma = g)
                rbf_svm.fit(X_train, Y_train)
                predictions = rbf_svm.predict(X_cv)
                accuracies[i, j] += (predictions == Y_cv).mean()
        print("new fold")
    accuracies = np.divide(accuracies, k)
    max = accuracies.max()
    min = accuracies.min()
    max_indices = np.asarray(np.unravel_index(accuracies.argmax(), accuracies.shape)) 
    min_indices = np.asarray(np.unravel_index(accuracies.argmin(), accuracies.shape)) 
    print("shape of max and min indices: {}, {}".format(max_indices.shape, min_indices.shape))
    print("\n\nMax value: {}, Min value: {}, Max params: c={}, gamma={}, Min params: c={}, gamma={}".format(max, min, C_range[max_indices[0]], gamma_range[max_indices[1]], C_range[min_indices[0]], gamma_range[min_indices[1]])) 
    plotter.plot2d_hist('g', 'C', gamma_range, C_range, accuracies)
    return max_indices

def evaluate_using_rbf_svm(X_train, Y_train, X_test, Y_test, c, g):
    rbf_svm = SVC(C=c, gamma = g)
    rbf_svm.fit(X_train, Y_train)
    train_predictions = rbf_svm.predict(X_train)
    test_predictions = rbf_svm.predict(X_test)
    print("RBF-SVM: train: {}, test: {}".format((train_predictions==Y_train).mean(), (test_predictions==Y_test).mean()))
    return train_predictions, test_predictions

def reduce_dimensions_using_lasso_T(X_train, X_test, Y_train,c):
    svm = LinearSVC(C=c, penalty="l1", dual=False)
    svm.fit(X_train, Y_train)
    model = SelectFromModel(svm, prefit=True)
    return model.transform(X_train), model.transform(X_test)

def get_subset_for_cv(samples_per_class, train, target):
    sampled_indices = np.zeros((1, 20*samples_per_class))
    for i in np.arange(20):
        indices = np.asarray((np.where(target == i)))[0, 0:samples_per_class]
        start_index = samples_per_class * i
        sampled_indices[0, start_index:(start_index+samples_per_class)] = indices
    sampled_indices = sampled_indices[0].astype(int)
    return train[sampled_indices,:], target[sampled_indices]

def reduce_dimensions_using_lasso(X_train, Y_train,c):
    svm = LinearSVC(C=c, penalty="l1", dual=False)
    svm.fit(X_train, Y_train)
    model = SelectFromModel(svm, prefit=True)
    return model.transform(X_train)

def kfold_cv_rbf_svm_w_lasso(k, train, target):
    count = 0
    kf = KFold(n_splits = k, shuffle = True)
    c_lasso = np.array([0.1, 0.5, 1,5, 10, 20])
    gamma_range = np.array([1e-3, 1e-2, 1e-1])
    C_range = np.array([1000, 5000, 10000])
    accuracies = np.zeros((c_lasso.shape[0],C_range.shape[0], gamma_range.shape[0]))
    reduced_dims_avg = np.zeros(c_lasso.shape)
    for a, c_las in enumerate(c_lasso):
        reduced_train = reduce_dimensions_using_lasso(train, target, c_las)
        reduced_dims_avg[a] = reduced_train.shape[1]
        for train_index, cv_index in kf.split(train):
            print("new fold")
            X_train, X_cv = reduced_train[train_index], reduced_train[cv_index]
            Y_train, Y_cv = target[train_index], target[cv_index]
            for i, c in enumerate(C_range):
                for j, g in enumerate(gamma_range):
                    count+=1
                    print("Computation: {} of {}".format(count, (c_lasso.shape[0]*k*C_range.shape[0]*gamma_range.shape[0])))
                    rbf_svm = SVC(C=c, gamma = g)
                    rbf_svm.fit(X_train, Y_train)
                    predictions = rbf_svm.predict(X_cv)
                    accuracies[a, i, j] += (predictions == Y_cv).mean()
    accuracies = np.divide(accuracies, k)
    plotter.plot3d_hist('g', 'C', gamma_range, C_range, accuracies, 'cl', c_lasso)
    plotter.plot2d_hist('g', 'C', gamma_range, C_range, reduced_dims_avg)
   # max = accuracies.max()
   # min = accuracies.min()
   # max_indices = np.asarray(np.unravel_index(accuracies.argmax(), accuracies.shape))
   # min_indices = np.asarray(np.unravel_index(accuracies.argmin(), accuracies.shape))
   # print("shape of max and min indices: {}, {}".format(max_indices.shape, min_indices.shape))
   # print("\n\nMax value: {}, Min value: {}, Max params: c={}, gamma={}, Min params: c={}, gamma={}".format(max, min, C_range[max_indices[0]], gamma_range[max_indices[1]], C_range[min_indices[0]], gamma_range[min_indices[1]]))

def evaluate_using_logistic_regression(X_train, Y_train, X_test, Y_test, c):
    LR = linear_model.LogisticRegression(penalty = 'l2', C=c, solver = 'saga', multi_class = 'multinomial')
    LR.fit(X_train, Y_train)
    train_predictions = LR.predict(X_train)
    test_predictions = LR.predict(X_test)
    print("Logistic Regression using 'l2' penalty, and multinomial: train: {}, test: {}".format((train_predictions==Y_train).mean(), (test_predictions==Y_test).mean()))
    return train_predictions, test_predictions

    '''
    LR = linear_model.LogisticRegression(penalty = 'l1', C=c, solver = 'saga', multi_class = 'multinomial')
    LR.fit(X_train, Y_train)
    train_predictions = LR.predict(X_train)
    test_predictions = LR.predict(X_test)
    print("Logistic Regression using 'l1' penalty, and multinomial: train: {}, test: {}".format((train_predictions==Y_train).mean(), (test_predictions==Y_test).mean()))

    LR = linear_model.LogisticRegression(penalty = 'l2', C=c, solver = 'saga', multi_class = 'ovr')
    LR.fit(X_train, Y_train)
    train_predictions = LR.predict(X_train)
    test_predictions = LR.predict(X_test)
    print("Logistic Regression using 'l2' penalty, and ovr: train: {}, test: {}".format((train_predictions==Y_train).mean(), (test_predictions==Y_test).mean()))
    
    LR = linear_model.LogisticRegression(penalty = 'l1', C=c, solver = 'saga', multi_class = 'ovr')
    LR.fit(X_train, Y_train)
    train_predictions = LR.predict(X_train)
    test_predictions = LR.predict(X_test)
    print("Logistic Regression using 'l1' penalty, and ovr: train: {}, test: {}".format((train_predictions==Y_train).mean(), (test_predictions==Y_test).mean()))
'''

def kfold_cv_LR(k, train, target):
    kf = KFold(n_splits = k, shuffle=True)
    c_array = np.array([25, 50, 75])
    accuracies = np.zeros((k,c_array.shape[0]))
    k_idx = 0
    count = 0
    for train_index, cv_index in kf.split(train):
        print("new fold")
        X_train, X_cv = train[train_index], train[cv_index]
        Y_train, Y_cv = target[train_index], target[cv_index]
        for i, c in enumerate(c_array): 
            print("Computation: {} of {}".format(count, (c_array.shape[0]*k)))
            LR = linear_model.LogisticRegression(penalty = 'l2', C=c, solver = 'saga', multi_class = 'multinomial')
            LR.fit(X_train, Y_train)
            prediction = LR.predict(X_cv)
            accuracies[k_idx,i]=(Y_cv == prediction).mean() 
            count+=1
        k_idx+=1
    means = np.mean(accuracies, axis=0)
    print("min: {} for {}, max: {} for {}".format(means.min(), c_array[np.argmin(means)], means.max(), c_array[np.argmax(means)]))
    max_idx = np.argmax(means)
    return c_array[max_idx]

def compute_confusion_matrix(number_of_classes, predictions, targets):
    confusion = np.zeros((number_of_classes, number_of_classes))
    for i in np.arange(number_of_classes):
        indices = np.where(targets == i)[0]      
        tmp_predictions = predictions[indices]
        for j in np.arange(number_of_classes):
            count = (np.where(tmp_predictions == j)[0]).shape[0]
            confusion[j,i] = count
    return confusion

def determine_confused_classes(confusion):
    confusion = confusion/(np.sum(confusion, axis = 0))
    np.fill_diagonal(confusion, 0)
    sum = (confusion + confusion.T)
    most_confused = np.unravel_index(sum.argmax(), sum.shape)
    print("Most Confused Classes are: {}".format(most_confused))

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_idf, test_idf, feature_names_idf = tf_idf_features(train_data, test_data)
    train_target = train_data.target
    test_target = test_data.target
    # ---------------------------- 1. LOGISTIC REGRESSION CLASSIFIER ------------------------------------------------------
   
    # *** 1.1.1. Standard Logistic Regression ***
    #evaluate_using_logistic_regression(train_idf, train_target, test_idf, test_target, 1)
   
    # *** 1.1.2. Cross Validate to find optimal C for Logistic Regression  ***
    #c = kfold_cv_LR(5, train_idf, train_target)
    evaluate_using_logistic_regression(train_idf, train_target, test_idf, test_target, 50)

    # --------------------------------------- 2. LINEAR SVM CLASSIFIER -----------------------------------------------------

    # *** 1.2.1.Standard Linear SVM ***
    #evaluate_using_linear_svm(train_idf, train_target, test_idf, test_target, 1)   
   
    # *** 1.2.2. Cross validate to find optimal C for linear SVM: ***
    #c = kfold_cv_lin_svm(5, train_idf, train_target)
    train_linSVM_preds, test_linSVM_preds = evaluate_using_linear_svm(train_idf, train_target, test_idf, test_target, 0.5)

    # *** 1.2.3. Linear SVM w Lasso regularization to determine relevant feature ***
    #c_lasso = kfold_cv_lin_svm_lasso(5, train_idf, train_target)
    #reduced_train, reduced_test = evaluate_using_linear_lasso_svm(train_idf, train_target, test_idf, test_target, 1.25)
    #print("Original Feature dimension: {}, Feature dimension after using Lasso: {}".format(train_idf.shape[1], reduced_train.shape[1]))    
   
    # ------------------------------ 3. NON-LINEAR RBF-KERNEL BASED SVM CLASSIFIER -----------------------------------------------

    # *** 1.3.1. Non-Linear SVM using RBF Kernel with original 'idf' features ***
    # CV taking too long on entire dataset, therefore use subset of data for CV
    #train_subset, target_subset = get_subset_for_cv(300, train_idf, train_target)
    #rbf_hyperparams = kfold_cv_rbf_svm(5, train_subset, target_subset)
    evaluate_using_rbf_svm(train_idf, train_target, test_idf, test_target, 10, 0.3)

    # *** 1.3.2. Non-Linear SVM using RBF Kernel with reduced features from 1.2.3 ***.
    #rbf_params = kfold_cv_rbf_svm(5, reduced_train, train_target)
    #print("The optimized parameters are: {}".format(rbf_params))
    #evaluate_using_rbf_svm(reduced_train, train_target, reduced_test, test_target, 10, 0.1)
     
    # *** TINKERING AROUND: SEEING IF REDUCING FEATURES EVEN MORE CAN IMPROVE PERFORMANCE OF RBF-SVM ***
    #kfold_cv_rbf_svm_w_lasso(2, train_idf, train_target)
    #reduced_train, reduced_test = reduce_dimensions_using_lasso_T(train_idf, test_idf, train_target, 0.2)
    #print("Original Feature dimension: {}, Feature dimension after using Lasso: {}".format(train_idf.shape[1], reduced_train.shape[1]))    
    #rbf_params = kfold_cv_rbf_svm(5, reduced_train, train_target)
    #evaluate_using_rbf_svm(reduced_train, train_target, reduced_test, test_target, 10, 1)

   # --------------------------- 4. BERNOULLI NAIVE BAYES CLASSIFIER ----------------------------------------------------------
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    # *** CONFUSION MATRIX ANALYSIS ***
    C = compute_confusion_matrix(20, test_linSVM_preds, test_target)
    plotter.plot_matrix(C)
    determine_confused_classes(C)
