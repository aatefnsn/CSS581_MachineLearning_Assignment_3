import scipy
import numpy as np
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix

x = scipy.io.loadmat('heart.mat')
print(x['dat'])
print(x['dat'].shape)
print(x['label'])
print(x['label'].shape)
dat = x['dat']
print(dat.shape)
#dat = np.reshape(dat,(270,13))
label = x['label']
print(label.shape)
#x.info()

def featureSubsetScore(X, y):
    trainScoreMean = 0.0
    testScoreMean = 0.0
    #print("Hello from featureSubsetScore")
    #print(X.shape)
    #print(y.shape)
    #if (X.shape[0] == y.shape[0]):
        #print("there are test set class labels equivalent to the training set ")
    for i in range(1, 11):
        #print(i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.259, random_state = i)
        #print(X_train.shape)
        #print(X_test.shape)
        #print(y_train.shape)
        #print(y_test.shape)
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        # y_pred = logreg.predict(X_test)
        # y_pred_train = logreg.predict(X_train)
        # print("y_pred is ", y_pred)
        # print(confusion_matrix(y_test, y_pred))
        # print(confusion_matrix(y_train, y_pred_train))
        trainScoreMean += logreg.score(X_train, y_train)
        testScoreMean += logreg.score(X_test, y_test)
        #print('Accuracy of Logistic regression classifier on training set: {:.5f}'.format(logreg.score(X_train, y_train)))
        #print('Accuracy of Logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test, y_test)))
    #print("trainScoreMean is ", trainScoreMean/10)
    #print("testScoreMean is ", testScoreMean/10)
    return (round((trainScoreMean/10),2),round((testScoreMean/10),2))

def main():
    #print("Hello from main!")
    #X = x['dat']
    #y = x['label']
    X = dat[:, :]
    y = label
    #y=label(:,1)
    #print("X is \n ", X)
    #print("from main \n",dat.shape)
    #print("from main \n",label.shape)
    #print(dat[1,1:5])
    trainScoreMean, testScoreMean = featureSubsetScore(X, y)
    print ("trainScoreMean = \n",trainScoreMean)
    print("testScoreMean = \n",testScoreMean)
    X = dat[:, [1, 3, 5]]
    #y = label
    trainScoreMean, testScoreMean = featureSubsetScore(X, y)
    print("trainScoreMean = \n", trainScoreMean)
    print("testScoreMean = \n", testScoreMean)

if __name__ == "__main__":
    main()
