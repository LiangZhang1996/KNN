import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from math import factorial
from numpy.linalg import norm
#from scipy.special import comb

## Read the data from the four files
# read a data file
def read_data(fname):
    f = open(fname,'r')
    data=f.readlines()
    f.close()
    X,Y=[],[]
    for row in data:
        row = row.replace('\n','')
        row = row.split(' ')        
        Y.append(row[0])
        X.append([float(x) for x in row[1:]])
    return X,Y
# prepare our data set with read in the files
def prepare_dataset():
    fnames = ['animals','countries','fruits','veggies']
    # dataX, dataY is the data we will use
    X,_ = read_data(fnames[0])
    dataX, dataY = np.array(X), [fnames[0] for i in range(len(X))]
    for fname in fnames[1:4]:
        X,_ = read_data(fname)
        dataX = np.concatenate((dataX,X))
        dataY.extend([fname for i in range(len(X))])
    dataY = pd.Categorical(dataY).codes
    return dataX, dataY

# Functions on how to calculate Precision, Recall and F-measure
# define functions to calculate TP FP TN FN
def TP_FP(classes,clusters):
    # the input of classes and clusters must be numpy.array
    count = Counter(clusters)
    # the total pairs in the same cluster
    total=0
    for key in count.keys():
        if count[key]>=2:
            total = total+ mycomb(count[key],2)
    # calculate the TP
    TP=0
    for key in count.keys():
        idx1 = np.where(clusters==key)
        cla = classes[idx1]
        count1 = Counter(cla)
        for key1 in count1.keys():
            if count1[key1]>=2:
                TP = TP + mycomb(count1[key1],2)
        
    FP = total-TP
    return TP,FP
#  define the function to calculate FN TN
def FN_TN(classes,clusters):
    count = Counter(clusters)
    total = 0
    c = [v for v in count.values()]
    for i in range(0,len(c)-1):
        for j in range(i+1,len(c)):
            total = total+c[i]*c[j]
    FN=0
    count1 = Counter(classes)
    for key1 in count1.keys():
        idx = np.where(classes==key1)
        cla = clusters[idx]
        count2 = Counter(cla)
        c2 = [v for v in count2.values()]
        for i in range(0,len(c2)-1):
            for j in range(i+1,len(c2)):
                FN = FN+c2[i]*c2[j]    
    TN=total-FN
    return FN,TN
# define functions to calculate Precision, Recall, and F-measure
def PRF(classes,clusters):
    TP,FP = TP_FP(classes,clusters)
    FN,_ = FN_TN(classes,clusters)
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F = 2*P*R/(P+R)
    return P,R,F

# Euclidean Distance
def Euc_distance(x1, x2):
    # x1 x2 are two vectors
    return np.sqrt(sum((x1-x2)**2))
def init_Cent(data, k):
    # init the k centroids at the beginnning of k-means
    numS, d = data.shape
    centroids = np.zeros((k,d))
    for i in range(k):
        idx = int(np.random.uniform(0,numS))
        centroids[i,:]=data[idx,:]
    return centroids
# The k-means clustering with Euclidean Distance
def K_means(data,k):
    # k_means algorithm
    numS = data.shape[0]
    # store the cluster data
    cdata = np.zeros((numS,2))
    centroids = init_Cent(data, k)
    change = True
    while change:
        change = False
        for i in range(numS):
            # mini-distance used for compare distance, so set it a pretty large number
            mdist = 10000
            # define the cluster class
            midx = 0
            for j in range(k):
                dist = Euc_distance(centroids[j,:],data[i,:])
                # find the minimal distance 
                if dist< mdist:
                    mdist=dist
                    cdata[i,1] = mdist
                    midx = j
            if cdata[i,0]!= midx:
                change = True
                cdata[i,0]=midx
        for j in range(k):
            # re-calculate the centroid
            c_idx = np.nonzero(cdata[:,0]==j)
            inCluster = data[c_idx]
            centroids[j,:] = np.mean(inCluster, axis=0)
        
    return centroids, cdata   
# the manhattan distance
def Manh_distance(x1,x2):
    return np.sum(np.abs(x1-x2))

#k-means clustering with the manhattan distance
def K_means2(data,k):
    # k_means algorithm
    numS = data.shape[0]
    # store the cluster data
    cdata = np.zeros((numS,2))
    centroids = init_Cent(data, k)
    change = True
    while change:
        change = False
        for i in range(numS):
            # mini-distance used for compare distance, so set it a pretty large number
            mdist = 10000
            # define the cluster class
            midx = 0
            for j in range(k):
                dist = Manh_distance(centroids[j,:],data[i,:])
                # find the minimal distance 
                if dist< mdist:
                    mdist=dist
                    cdata[i,1] = mdist
                    midx = j
            if cdata[i,0]!= midx:
                change = True
                cdata[i,0]=midx
        for j in range(k):
            # re-calculate the centroid
            c_idx = np.nonzero(cdata[:,0]==j)
            inCluster = data[c_idx]
            centroids[j,:] = np.mean(inCluster, axis=0)
    # return the k-clusters' cnetroids and new class 
    return centroids, cdata 

# cosine similarity
# larger value means more similar, so we can't just use similarity as the distance
def Cos_similarity(x1,x2):
    return 1-sum(x1*x2)/(np.sqrt(sum(x1**2))*np.sqrt(sum(x2**2)))

#k-means clustering
def K_means3(data,k):
    # k_means algorithm
    numS = data.shape[0]
    # store the cluster data
    cdata = np.zeros((numS,2))
    centroids = init_Cent(data, k)
    change = True
    while change:
        change = False
        for i in range(numS):
            # mini-distance used for compare distance, so set it a pretty large number
            mdist = 10000
            # define the cluster class
            midx = 0
            for j in range(k):
                dist = Cos_similarity(centroids[j,:],data[i,:])
                # find the minimal distance 
                if dist< mdist:
                    mdist=dist
                    cdata[i,1] = mdist
                    midx = j
            if cdata[i,0]!= midx:
                change = True
                cdata[i,0]=midx
        for j in range(k):
            # re-calculate the centroid
            c_idx = np.nonzero(cdata[:,0]==j)
            inCluster = data[c_idx]
            centroids[j,:] = np.mean(inCluster, axis=0)        
    # return the k-clusters' cnetroids and new class 
    return centroids, cdata 

def plot_fig(P,R,F,save=False):
    K = np.arange(1,11)
    plt.figure()
    plt.plot(K,P,linewidth=2)
    plt.plot(K,R,linewidth=2)
    plt.plot(K,F,linewidth=2)
    plt.xlabel('K',fontsize=16)
    plt.ylabel('values')
    plt.legend(['Precision','Recall','F-score'],fontsize=13)
    if save:
        fname=input('Input the figure name you want to save:')
        plt.savefig(fname,dpi=300)
    plt.show()

def mycomb(n,k):
    assert n>=k
    if n==k:
        result = 1
    else:
        result = factorial(n)/(factorial(k)*factorial(n-k))
    return result
def mynorm(A):
    x = norm(A,ord=None,axis=1)
    normA = []
    for i in range(A.shape[0]):       
        normA.append(A[i]/x[i])
    return np.array(normA)


if __name__=='__main__':
    print(mycomb(4,3))