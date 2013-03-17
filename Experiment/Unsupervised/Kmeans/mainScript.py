__author__ = "Vincent Archambault-Bouffard"

# This script will train

import numpy
import patches
import preprocessing
import kmeans
import loadDataset

patchSize = 16
kMeansBatchSize = 1000


def unsupervisedTraining(X):
    """Unsupervised training using K-means

    Returns all information needed for training and testing"""
    print "Crop patches"
    p = numpy.concatenate([patches.crop_patches_grayscale(x, patchSize, 20) for x in X]).astype(numpy.float64)
    p = p.reshape((p.shape[0], -1))  # Flattens every patch

    print "Canonical preprocessing"
    p = preprocessing.contrastNormalization(p)
    p, dataMean, dataStd = preprocessing.standardScore(p)

    print "PCA Whitening"
    whitePatches, projectionMapping, inverseMapping = preprocessing.pca_whitening(p, 0.9)

    print "K-means"
    centroids = kmeans.kmeans(whitePatches, 400, 10, batchsize=kMeansBatchSize)

    return dataMean, dataStd, projectionMapping, inverseMapping, centroids


def applyUnsupervisedTraining(X, dataMean, dataStd, projectionMapping, centroids):
    features = []
    print "Input data", X.shape
    for i, img in enumerate(X):
        if i % 100 == 0:
            print i

        # Crop patches
        p = patches.crop_patches_grayscale(img, patchSize, 400).astype('float')
        p = p.reshape((p.shape[0], -1))  # Flattens the patch
        print "After crop", p.shape

        # Preprocess with unsupervised data
        p = preprocessing.apply_standardScore(p, dataMean, dataStd)
        p = preprocessing.apply_pca_whitening(p, projectionMapping)
        print "After whitening", p.shape

        # Extract the K-means features
        p = kmeans.assignTriangle(p, centroids)
        print "After assign ", p.shape
        p.mean(axis=1)
        features.append(p)

    return numpy.concatenate(features)

if __name__ == "__main__":
    print "Import dataset"
    trainX, trainY, testX = loadDataset.loadDataset()

    print "Unsupervised Training"
    dataMean, dataStd, projectionMapping, inverseMapping, centroids = unsupervisedTraining(trainX)

    print "Apply unsupervised training"
    features = applyUnsupervisedTraining(trainX[:1], dataMean, dataStd, projectionMapping, centroids)
    print features.sum(axis=1).mean()

    print "Train logistic regression"
