# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:31:23 2019

@author: Wladek
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from simple_neural_network.planar_utils import plot_decision_boundary
from simple_neural_network.NN_model import nn_model
from simple_neural_network.NN_function import forward_propagation


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    return A2 > 0.5


def load_extra_datasets():
    N = 200
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None,
                                                                  cov=0.7, n_samples=N, n_features=2, n_classes=2,
                                                                  shuffle=True, random_state=None)
    return gaussian_quantiles


gaussian_quantiles = load_extra_datasets()
X, Y = gaussian_quantiles
X, Y = X.T, Y.reshape(1, Y.shape[0])
# Visualize the data
Ydraw = Y[0]
plt.scatter(X[0, :], X[1, :], c=Ydraw, s=40, cmap=plt.cm.Spectral)
# %%
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.subplot()
# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      '% ' + "(percentage of correctly labelled datapoints)")
# %%
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=3, num_iterations=1000, print_cost=True)
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
