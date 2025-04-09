import scipy
import numpy as np
# import pandas as pd
from scipy.interpolate import rbf
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import math

np.set_printoptions(edgeitems=15, linewidth=300)
from operator import itemgetter
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import time
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
x = scipy.io.loadmat('USPS_all.mat')
print(x)
from random import sample

# x = sample(list(x), 1000, 0)
# print(x.shape)
data = x['fea']
label = x['gnd']
# print("Head is \n", data.dtype.names)
print("Data shape is ", data.shape)
print("Label shape is ", label.shape)
X = data
# X = X[1:1401,:]
print("X is \n", X)
y = label
# y= y[1:1401, :]
print("y is\n", y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21585, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=1)
print("X_train shape is ", X_train.shape)
print("y_train shape is ", y_train.shape)
print("X_test shape is ", X_test.shape)
print("y_test shape is ", y_test.shape)

clf = DecisionTreeClassifier()
start_time = time.time()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix is \n", confusion_matrix(y_test, y_pred))
print('Accuracy of Decision Tree classifier on training set: {:.5f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.5f}'.format(clf.score(X_test, y_test)))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(graph.create_png())

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5, splitter="best")
start_time = time.time()
clf_gini = clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix is \n", confusion_matrix(y_test, y_pred))
print('Accuracy of BEST GINI Decision Tree classifier on training set: {:.5f}'.format(clf_gini.score(X_train, y_train)))
print('Accuracy of BEST GINI Decision Tree classifier on test set: {:.5f}'.format(clf_gini.score(X_test, y_test)))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree_gini_best.png')
Image(graph.create_png())

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=7, min_samples_leaf=5,
                                  splitter="random")
start_time = time.time()
clf_gini = clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix is \n", confusion_matrix(y_test, y_pred))
print(
    'Accuracy of RANDOM GINI Decision Tree classifier on training set: {:.5f}'.format(clf_gini.score(X_train, y_train)))
print('Accuracy of RANDOM GINI Decision Tree classifier on test set: {:.5f}'.format(clf_gini.score(X_test, y_test)))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree_gini_random.png')
Image(graph.create_png())

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=10)
start_time = time.time()
clf_entropy = clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix is \n", confusion_matrix(y_test, y_pred))
print(
    'Accuracy of ENTROPY Decision Tree classifier on training set: {:.5f}'.format(clf_entropy.score(X_train, y_train)))
print('Accuracy of ENTROPY Decision Tree classifier on test set: {:.5f}'.format(clf_entropy.score(X_test, y_test)))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

from sklearn.svm import SVC

SVC = SVC(gamma='auto')
kernel = ['rbf', 'linear', 'poly']
C = [0.01, 0.1, 1, 10, 100]
decision_function_shape = ['ovr', 'ovo']
param_grid = dict(decision_function_shape=decision_function_shape, kernel=kernel, C=C)
grid = GridSearchCV(estimator=SVC, param_grid=param_grid, cv=3, n_jobs=-1)
start_time = time.time()
grid_result = grid.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

SVC.fit(X_train, y_train)
y_pred = SVC.predict(X_test)
print("y_pred is ", y_pred)
print(confusion_matrix(y_test, y_pred))
print('Accuracy of SVM classifier on training set: {:.5f}'.format(SVC.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.5f}'.format(SVC.score(X_test, y_test)))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

def euclidean_distance(row1, row2):# method to calculate the distance between two sample points, feature by feature
    distance = 0.0
    for i in range(len(row1) - 1):#subtract all features from each other except the last one as it contains the row label
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


def weighted_euclidean_distance(row1, row2):#same as above
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):#function to calculate the distance betwen a given point and all other points in the training dataset
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row) #calculatuing the distance for each point against all trainning rows
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])#sort the points by their distance in ascending
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0]) #append the neighboring points in a list up to the number of allowed negiboring points
    return neighbors


def weighted_get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = weighted_euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    # print("Neighbors sorted by distances: \n", distances)
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances)  # selecting the closest (num_neighbors) neighbors
    return neighbors


def predict_classification(train, test_row, num_neighbors): # method to predict classification based on majority voting
    neighbors = get_neighbors(train, test_row, num_neighbors)#get a list of neighboring points
    output_values = [row[-1] for row in neighbors] # get the labels of all neighboring points
    prediction = max(set(output_values), key=output_values.count) # selecting the most common label, majority voting and use it as the prediction result
    return prediction


def weighted_predict_classification(train, test_row, num_neighbors):
    neighbors = weighted_get_neighbors(train, test_row, num_neighbors) # get a list of neighboring points
    # print("Neighbors are: \n",neighbors)
    # print("Neighbors size is \n", np.size(neighbors))
    # print neighbors see how it looks like, should be the train rows, class, distance
    freq0 = 0  # weighted sum of group 0
    freq1 = 0  # weighted sum of group 1
    freq2 = 0  # weighted sum of group 2
    freq3 = 0  # weighted sum of group 3
    freq4 = 0  # weighted sum of group 4
    freq5 = 0  # weighted sum of group 5
    freq6 = 0  # weighted sum of group 6
    freq7 = 0  # weighted sum of group 7
    freq8 = 0  # weighted sum of group 8
    freq9 = 0  # weighted sum of group 9
    freq10 = 0  # weighted sum of group 10
    # for d in neighbors:
    for d in range(0, num_neighbors):
        # print("neighbor is", neighbors[0][d])
        # print("neighbor class is", neighbors[0][d][0][-1])
        if neighbors[0][d][0][-1] == 0: # calculate the weighted frequency of a class label by multiplying it by inverse the distance of this point
            freq0 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 1:
            freq1 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 2:
            freq2 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 3:
            freq3 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 4:
            freq4 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 5:
            freq5 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 6:
            freq6 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 7:
            freq7 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 8:
            freq8 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 9:
            freq9 += (1 / neighbors[0][d][1])
        if neighbors[0][d][0][-1] == 10:
            freq10 += (1 / neighbors[0][d][1])
    prediction = 0
    # if conditions to predict the class label based on the calculated weighted frequency
    if (
            freq0 > freq1 and freq0 > freq2 and freq0 > freq3 and freq0 > freq4 and freq0 > freq5 and freq0 > freq6 and
            freq0 > freq7 and freq0 > freq8 and freq0 > freq9 and freq0 > freq10):
        prediction = 0
    if (
            freq1 > freq0 and freq1 > freq2 and freq1 > freq3 and freq1 > freq4 and freq1 > freq5 and freq1 > freq6 and
            freq1 > freq7 and freq1 > freq8 and freq1 > freq9 and freq1 > freq10):
        prediction = 1
    if (
            freq2 > freq0 and freq2 > freq1 and freq2 > freq3 and freq2 > freq4 and freq2 > freq5 and freq2 > freq6 and
            freq2 > freq7 and freq2 > freq8 and freq2 > freq9 and freq2 > freq10):
        prediction = 2
    if (
            freq3 > freq0 and freq3 > freq1 and freq3 > freq2 and freq3 > freq4 and freq3 > freq5 and freq3 > freq6 and
            freq3 > freq7 and freq3 > freq8 and freq3 > freq9 and freq3 > freq10):
        prediction = 3
    if (
            freq4 > freq0 and freq4 > freq1 and freq4 > freq2 and freq4 > freq3 and freq4 > freq5 and freq4 > freq6 and
            freq4 > freq7 and freq4 > freq8 and freq4 > freq9 and freq4 > freq10):
        prediction = 4
    if (
            freq5 > freq0 and freq5 > freq1 and freq5 > freq2 and freq5 > freq3 and freq5 > freq4 and freq5 > freq6 and
            freq5 > freq7 and freq5 > freq8 and freq5 > freq9 and freq5 > freq10):
        prediction = 5
    if (
            freq6 > freq0 and freq6 > freq1 and freq6 > freq2 and freq6 > freq3 and freq6 > freq4 and freq6 > freq5 and
            freq6 > freq7 and freq6 > freq8 and freq6 > freq9 and freq6 > freq10):
        prediction = 6
    if (
            freq7 > freq0 and freq7 > freq1 and freq7 > freq2 and freq7 > freq3 and freq7 > freq4 and freq7 > freq5 and
            freq7 > freq6 and freq7 > freq8 and freq7 > freq9 and freq7 > freq10):
        prediction = 7
    if (
            freq8 > freq0 and freq8 > freq1 and freq8 > freq2 and freq8 > freq3 and freq8 > freq4 and freq8 > freq5 and
            freq8 > freq6 and freq8 > freq7 and freq8 > freq9 and freq8 > freq10):
        prediction = 8
    if (
            freq9 > freq0 and freq9 > freq1 and freq9 > freq2 and freq9 > freq3 and freq9 > freq4 and freq9 > freq5 and
            freq9 > freq6 and freq9 > freq7 and freq9 > freq8 and freq9 > freq10):
        prediction = 9
    if (
            freq10 > freq0 and freq10 > freq1 and freq10 > freq2 and freq10 > freq3 and freq10 > freq4 and
            freq10 > freq5 and freq10 > freq6 and freq10 > freq7 and freq10 > freq8 and freq10 > freq9):
        prediction = 10
    return prediction


def k_nearest_neighbors(train, test, num_neighbors): # method to calculate the class label of a each row in test set by
    # comparing each test point with all the training set points
    predictions = list()
    i = 0
    for row in test:
        i = i + 1
        # print(i)
        # print("row is ", row)
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions


def weighted_k_nearest_neighbors(train, test, num_neighbors): # method to calculate the class label of a each row in
    # test set by comparing each test point with all the training set points and giving weights to the closest class labels
    predictions = list()
    i = 0
    for row in test:
        i = i + 1
        # print(i)
        # print("row is ", row)
        output = weighted_predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions


X = X[1:1401, :]
y = y[1:1401, :]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21585, random_state=1)

train_set = np.append(X_train, y_train, 1)
print("train set shape is ", train_set.shape)
test_set = np.append(X_test, y_test, 1)
print("test set shape is", test_set.shape)

start_time = time.time()
predicted = k_nearest_neighbors(train_set, test_set, 1)
print("Predicted is \n", predicted)
print("k =1")
print("Confusion Matrix is \n", confusion_matrix(y_test, predicted))
print("Accuracy Score :", accuracy_score(y_test, predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

start_time = time.time()
predicted = k_nearest_neighbors(train_set, test_set, 5)
print("Predicted is \n", predicted)
print("k =5")
print("Confusion Matrix is \n", confusion_matrix(y_test, predicted))
print("Accuracy Score :", accuracy_score(y_test, predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

start_time = time.time()
predicted = k_nearest_neighbors(train_set, test_set, 6)
print("Predicted is \n", predicted)
print("k =6")
print("Confusion Matrix is \n", confusion_matrix(y_test, predicted))
print("Accuracy Score :", accuracy_score(y_test, predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

start_time = time.time()
predicted = k_nearest_neighbors(train_set, test_set, 10)
print("Predicted is \n", predicted)
print("k =10")
print("Confusion Matrix is \n", confusion_matrix(y_test, predicted))
print("Accuracy Score :", accuracy_score(y_test, predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

start_time = time.time()
weighted_predicted = weighted_k_nearest_neighbors(train_set, test_set, 1)
print("weighted Predicted is \n", weighted_predicted)
print("k =1")
print("Confusion Matrix is \n", confusion_matrix(y_test, weighted_predicted))
print("Accuracy Score :", accuracy_score(y_test, weighted_predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

start_time = time.time()
weighted_predicted = weighted_k_nearest_neighbors(train_set, test_set, 5)
print("weighted Predicted is \n", weighted_predicted)
print("k =5")
print("Confusion Matrix is \n", confusion_matrix(y_test, weighted_predicted))
print("Accuracy Score :", accuracy_score(y_test, weighted_predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

start_time = time.time()
weighted_predicted = weighted_k_nearest_neighbors(train_set, test_set, 6)
print("weighted Predicted is \n", weighted_predicted)
print("k =6")
print("Confusion Matrix is \n", confusion_matrix(y_test, weighted_predicted))
print("Accuracy Score :", accuracy_score(y_test, weighted_predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')

start_time = time.time()
weighted_predicted = weighted_k_nearest_neighbors(train_set, test_set, 10)
print("weighted Predicted is \n", weighted_predicted)
print("k =10")
print("Confusion Matrix is \n", confusion_matrix(y_test, weighted_predicted))
print("Accuracy Score :", accuracy_score(y_test, weighted_predicted))
print("Execution time: " + str((time.time() - start_time)) + ' seconds')
