import scipy
import numpy as np
from Task3 import featureSubsetScore
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

np.set_printoptions(edgeitems=15, linewidth=300)
from scipy.io import loadmat
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from numpy import mean
# from numpy import cov
# from numpy.linalg import eig
# from sklearn.decomposition import PCA
# from matplotlib import pyplot as plt
# from sklearn.metrics import mean_squared_error, confusion_matrix

x = scipy.io.loadmat('heart.mat')
#print(x['dat'])
#print(x['dat'].shape)
#print(x['label'])
#print(x['label'].shape)
dat = x['dat']
dat = dat.astype(int)
label = x['label']
#print(dat)
#print(dat.shape)
#print("names are: \n", dat.dtype.names)


#k = 9
import random
for k in range(1, 14):
    ScoreBest = 0
    FSel = []
    SEED = 1
    #myListList = np.zeros((1000,1))
    SequenceList = []
    for i in range(1, 1001):
    #for i in range(1, 2):
        #random.seed(SEED)
        random.seed(i)
        Sequence = np.random.randint(2, size=13)
        #if Sequence.sum() > k or Sequence.sum() == 0:
        if Sequence.sum() != k:
            while True:
                Sequence = np.random.randint(2, size=13)
                #if Sequence.sum() > k or Sequence.sum() == 0:
                if Sequence.sum() != k:
                    continue
                break
        #print("Sequence is ", Sequence)
        dat_work = dat
        index = 0
        for j in range (0,13):
            #print ("j is ", myList[j])
            if Sequence[j] == 0:
                #print ("j is ", j)
                dat_work = np.delete(dat_work, [index], axis=1) ## removing the columns that corresepond to 0 in the sequence
                index = index - 1
                #continue
            index = index + 1
        SequenceList.append(Sequence)
        trainScoreMean, testScoreMean = featureSubsetScore(dat_work, label)
        #print("trainScoreMean = \n", trainScoreMean)
        #print("testScoreMean = \n", testScoreMean)
        if testScoreMean > ScoreBest:
            ScoreBest = testScoreMean
            FSel = Sequence
        #print("SEED is: ", SEED)
        #SEED += 1
    print("k is: ", k)
    print("Score Best is: ", ScoreBest)
    print("Best Sequence is ", FSel)
    SequenceList = np.asarray(SequenceList)
#print(SequenceList)
#print(dat_work)
#print(SequenceList.shape)