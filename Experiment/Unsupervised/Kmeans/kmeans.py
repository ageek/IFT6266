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
    index = numpy.unique(numpy.random.randint(0, X.shape[0], k))
    while index.shape[0] != k:
        missing = k - index.shape[0]
        new = numpy.random.randint(0, X.shape[0], missing)
        index = numpy.unique(numpy.concatenate([index, new]))
    return X[index, :]


def assign(X, W):
    """
    Assigns each vector to the nearest centroids

    The data in X must be row wise
    """
    X2 = (X ** 2).sum(1)[:, None]
    W2 = (W ** 2).sum(1)[:, None]
    D = -2 * numpy.dot(W, X.T) + W2 + X2.T
    return (D == D.min(0)[None, :]).astype(int)


def assignTriangle(X, W):
    """
    Assigns each vectors to the centroids that are closer than the mean centroid

    It is a softer version of assign but still yields sparse outputs
    Comes from "An Analysis of Single Layer Networks in Unsupervised Feature Learning" by Adam Coates

    The data in X must be row wise
    """
    X2 = (X ** 2).sum(1)[:, None]
    W2 = (W ** 2).sum(1)[:, None]
    D = -2 * numpy.dot(W, X.T) + W2 + X2.T
    D **= 0.5
    D = D.average(axis=0) - D
    D[D < 0] = 0
    return D