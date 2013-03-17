__author__ = "Vincent Archambault-Bouffard"

import pylab
import numpy
import patches
import imageUtils
import preprocessing
import kmeans


def import_dataset():
    """Imports the dataset"""
    trainX = pylab.loadtxt("cifar_mini_images_train.txt").reshape(2000, 3, 32, 32).transpose(0, 2, 3, 1)
    trainY = pylab.loadtxt("cifar_mini_labels_train.txt").reshape(2000, 10)
    testX = pylab.loadtxt("cifar_mini_images_test.txt").reshape(2000, 3, 32, 32).transpose(0, 2, 3, 1)
    testY = pylab.loadtxt("cifar_mini_labels_test.txt").reshape(2000, 10)

    return trainX, trainY, testX, testY


def unsupervisedTraining(X):
    """Unsupervised training using K-means

    Returns all information needed for training and testing"""
    print "Crop patches"
    p = numpy.concatenate([patches.crop_patches_color(x, 10, 6) for x in X]).astype(numpy.float64)
    p = p.reshape((p.shape[0], -1))  # Flattens every patch

    print "Canonical preprocessing"
    p, dcMean, contrastStd, dataMean, dataStd = preprocessing.canonical_preprocessing(p)

    print "PCA Whitening"
    whitePatches, projectionMapping, inverseMapping = preprocessing.pca_whitening(p, 0.9)

    print "K-means"
    centroids = kmeans.kmeans(whitePatches, 100, 10)

    return dcMean, contrastStd, dataMean, dataStd, projectionMapping, inverseMapping, centroids


if __name__ == "__main__":
    print "Import dataset"
    trainX, trainY, testX, testY = import_dataset()

    print "Unsupervised Training"
    dcMean, contrastStd, dataMean, dataStd, projectionMapping, inverseMapping, centroids = unsupervisedTraining(trainX)

    print "Training classifier"
    for i, img in enumerate(trainX):
        if i % 100 == 0:
            print i

        # Crop patches
        p = patches.crop_patches_color(img, 10, 400).astype(numpy.float64)
        p = p.reshape((p.shape[0], -1))  # Flattens the patch

        # Preprocess with unsupervised data
        p = preprocessing.apply_canonical_preprocessing(p, dcMean, contrastStd, dataMean, dataStd)
        p = preprocessing.apply_pca_whitening(p, projectionMapping)

        # Extract the K-means features
        features = kmeans.assignTriangle(p, centroids)

        # Compute mean features from by quadrants
