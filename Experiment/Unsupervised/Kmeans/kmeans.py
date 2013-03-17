# coding=utf-8
__author__ = ['Roland Memisevic', 'Vincent Archambault-Bouffard']

import numpy


def kmeans(X, k, numepochs, Winit=None, learningrate=0.01, batchsize=100, verbose=True):
    """
    Kmeans algorithm
    Returns the k centroids

    The data in X must be row wise
    """
    if Winit is None:
        W = _randomCenter(X, k)
    else:
        W = Winit

    X2 = (X ** 2).sum(1)[:, None]
    for epoch in range(numepochs):
        for i in range(0, X.shape[0], batchsize):
            W2 = (W ** 2).sum(1)[:, None]
            D = -2 * numpy.dot(W, X[i:i + batchsize, :].T) + W2 + X2[i:i + batchsize].T
            S = (D == D.min(0)[None, :]).astype("float")
            clustersums = numpy.dot(S, X[i:i + batchsize, :])
            pointspercluster = S.sum(1)[:, None]
            W += learningrate * (clustersums - pointspercluster * W)
        if verbose:
            cost = D.min(0).sum()
            print "epoch", epoch + 1, "of", numepochs, " cost: ", cost

    return W


def _randomCenter(X, k):
    """
    Set the initial centroids as randomly chosen data points
    """
    R = numpy.random.permutation(X.shape[0])
    init = X[R, :]
    init = init[:k, :]
    return init


def assign(X, W):
    """
    Assigns each vector to the nearest centroids

    The data in X must be row wise
    Centroid must be row wise
    Return shape = (X.shape[0],W.shape[0])
    """
    X2 = (X ** 2).sum(1)[:, None]
    W2 = (W ** 2).sum(1)[:, None]
    D = -2 * numpy.dot(W, X.T) + W2 + X2.T
    D = (D == D.min(0)[None, :]).astype(int)
    return D.T


def assignTriangle(X, W):
    """
    Assigns each vectors to the centroids that are closer than the mean centroid

    It is a softer version of assign but still yields sparse outputs
    Comes from "An Analysis of Single Layer Networks in Unsupervised Feature Learning" by Adam Coates

    The data in X must be row wise
    Centroid must be row wise
    Return shape = (X.shape[0],W.shape[0])
    """
    X2 = (X ** 2).sum(1)[:, None]
    W2 = (W ** 2).sum(1)[:, None]
    D = -2 * numpy.dot(W, X.T) + W2 + X2.T
    D **= 0.5
    D = D.mean(axis=0) - D
    D[D < 0] = 0
    return D.T

def randomCenter(data,nbCenter):
    index = numpy.unique(numpy.random.randint(0,data.shape[0],nbCenter))
    while index.shape[0] != nbCenter:
        missing = nbCenter - index.shape[0]
        new = numpy.random.randint(0,data.shape[0],missing)
        index = numpy.unique(numpy.concatenate([index,new]))
    return index


def kmeans_batch(data,centroidsNumber, center = None):
    dim = data.shape[1]
    nObs = data.shape[0]

    if center is None:
        center = data[randomCenter(data,centroidsNumber),:]

    nCenters = center.shape[0]

    i = 0
    newCenter = numpy.zeros((nCenters,dim))
    nearest = numpy.zeros(nObs)
    while True:
        i = i+1

        # Trouver le centre le plus proche
        xSquare = numpy.sum(data**2,axis=1)
        cSquare = numpy.sum(center**2,axis=1)
        crossTerm = numpy.dot(data,center.T)
        distance = ((-2 * crossTerm).T + xSquare).T + cSquare
        nearest = numpy.argmin(distance,axis=1)

        # Mis à jour du centre
        for j in range(nCenters):
            count = numpy.sum(nearest == j)
            if count >0:
                newCenter[j,:] = numpy.sum(data[nearest == j,:],axis=0) / count
            else:
                newCenter[j,:] = center[j,:]

        if numpy.all(newCenter == center):
            return center
        center = newCenter

