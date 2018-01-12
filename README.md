# PythonML

Applying Machine Learning algorithms using the Python ecosystem (Numpy, Scikit-learn, Pandas, SciPy, Matplotlib) to various datasets.

The four different folders contain code being applied to a different dataset:

1) 20NewsGroupClassification - A comparison of the performance of different classifiers when applied to the 20NewsGroup dataset: http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
	- 'classifier_comparison.py': A Bernoulli Naive Bayes classifier is used as the baseline model, and its performance is compared to:
		1) Logistic Regression Classifier
		2) Linear SVM Classifier
		3) RBF Kernel based SVM Classifier

2) BostonHousingRegression - Regression algorithms for prediction of housing price using Boston housing dataset: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
	- 'linear_regression.py': Trains and tests a linear regression model on the Boston housing dataset
	- 'locally_reweighted_regression.py': Trains and tests a locally reweighted regression model on the Boston housing dataset

3) HandwrittenDigitClassification - Classification algorithms for labeling 8x8 pixel images of handwritten digits: Dataset provided in 'data' folder
	- 'gaussian_bayes.py': Trains and tests a Gaussian Bayes classifier on the provided dataset
	- 'knn.py': Trains and tests K-nearest neighbours classifiers on the provided dataset
	- 'naive_bayes.py': Trains and tests a naive bayes model on the provided dataset

4) MNISTDigitClassification - Classification algorithms when applied to MNIST handwritten digit dataset for binary classification of 4 vs 9 (hardest 1 vs 1 pair): http://yann.lecun.com/exdb/mnist/
	- 'sgd-based_svm.py': Trains using stochastic gradient descent with momentum a linear SVM model and tests it using MNIST dataset
	
Technologies used: Python, NumPy, SciPy, Pandas, SciKit-learn, Matplotlib
