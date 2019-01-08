from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]] #center
cov = [[1, 0], [0, 1]]

N = 500
#create data with means is above
#cov is ......
#N is numbers of data
x0 = np.random.multivariate_normal(means[0], cov, N)
x1 = np.random.multivariate_normal(means[1], cov, N)
x2 = np.random.multivariate_normal(means[2], cov, N)

#X is data
X = np.concatenate((x0, x1, x2), axis = 0)
K = 3

# print([1]*N + [2]*N)
#this is label of data
original_label = np.asarray([0]*N + [1]*N + [2]*N).T 

def kmean_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    # print(X0[:, 2], X0[:, 1])

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()


# print(cov.shape)

#khoi tao cac center ban dau
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

#gan nhan moi cho cac diem khi biet centers
def kmean_assign_labels(X, centers):
    #calculate distances btw data and centers
    D = cdist(X, centers)
    #return index of the closest center
    return np.argmin(D, axis=1)
#update centers after calculate distance of data
def kmean_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    
    for k in range(K):
        #collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]

        # print(np.mean(Xk, axis=0))
        # print("-------------------------")
        
        #take average
        centers[k, :] = np.mean(Xk, axis = 0)
        
    
    return centers

#check stop condition
def has_converged(centers, new_centers):
    #return True if tow sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
    set([tuple(a) for a in new_centers]))

x = np.asarray([1, 2, 3])

# print(X[x])
# print(kmeans_init_centers(X, K))

def kmeans(X, K):
  
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    # print(centers)
    for i in range(100):
        print(centers[-1])
        print("-------------------------")
        labels.append(kmean_assign_labels(X, centers[-1]))
        new_centers = kmean_update_centers(X, labels[-1], K)
        print(new_centers)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)


(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmean_display(X, labels[-1])
