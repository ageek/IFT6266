__author__ = "Vincent Archambault-Bouffard"

# This script will train

import os
import numpy
import patches
import preprocessing
import kmeans
import loadDataset
import logreg

patchSize = 6
validationSize = 500

kMeansBatchSize = 100
kMeansIter = 30
centroidsNumber = 1000

logIter = 1500
logWeightDecay = 0.001


def unsupervisedTraining(X):
    """Unsupervised training using K-means

    Returns all information needed for training and testing"""
    print "Crop patches"
    p = numpy.concatenate([patches.crop_patches_grayscale(x, patchSize, 40) for x in X]).astype(numpy.float64)
    p = p.reshape((p.shape[0], -1))  # Flattens every patch

    print "Canonical preprocessing"
    p = preprocessing.contrastNormalization(p)
    p, dataMean, dataStd = preprocessing.standardScore(p)

    print "PCA Whitening"
    whitePatches, projectionMapping, inverseMapping = preprocessing.pca_whitening(p, 0.9)

    print "K-means"
    centroids = kmeans.kmeans(whitePatches, centroidsNumber, kMeansIter, batchsize=kMeansBatchSize)
    #centroids = kmeans.kmeans_batch(whitePatches, centroidsNumber)

    return dataMean, dataStd, projectionMapping, inverseMapping, centroids


def applyUnsupervisedTraining(X, dataMean, dataStd, projectionMapping, centroids):
    features = []
    #print "Input data", X.shape
    for i, img in enumerate(X):
        if i % 100 == 0:
            pass  # print i

        # Crop patches
        p = patches.crop_patches_grayscale_everywhere(img, patchSize).astype('float')
        p = p.reshape((p.shape[0], -1))  # Flattens the patch
        #print "After crop", p.shape

        # Preprocess with unsupervised data
        p = preprocessing.contrastNormalization(p)
        p = preprocessing.apply_standardScore(p, dataMean, dataStd)
        p = preprocessing.apply_pca_whitening(p, projectionMapping)
        #print "After whitening", p.shape

        # Extract the K-means features
        #print "Centroid ", centroids.shape
        #p = kmeans.assignTriangle(p, centroids)
        p = kmeans.assign(p, centroids)
        #print "After assign ", p.shape
        #print "Average activation"
        #p2 = p.sum(axis=1)
        #p2 = p2.mean()
        #print p2
        p = p.sum(axis=0)[None, :]
        #print p
        features.append(p)

    return numpy.concatenate(features)

if __name__ == "__main__":
    print "Import dataset"
    trainX, trainY, testX = loadDataset.loadDataset()
    # Shuffle the data set
    R = numpy.random.permutation(trainX.shape[0])
    trainX = trainX[R, :, :]
    trainY = trainY[R, :]

    # Split train in train and validation
    cut = trainX.shape[0] - validationSize
    validX = trainX[cut:, :]
    validY = trainY[cut:, :]
    trainX = trainX[:cut, :]
    trainY = trainY[:cut, :]
    print "Train", trainX.shape
    print "Valid", validX.shape
    print "Test", testX.shape

    print "Unsupervised Training"
    dataMean, dataStd, projectionMapping, inverseMapping, centroids = unsupervisedTraining(trainX)

    print "Apply unsupervised training"
    features = applyUnsupervisedTraining(trainX, dataMean, dataStd, projectionMapping, centroids)
    print features.shape

    print "Train logistic regression"
    logR = logreg.Logreg(7, centroidsNumber)
    logR.train(features.T, trainY.T, logWeightDecay, logIter, verbose=False)

    print "Computing valid %"
    validFeatures = applyUnsupervisedTraining(validX, dataMean, dataStd, projectionMapping, centroids)
    print validFeatures.shape
    validClassified = logR.classify(validFeatures.T).T
    print validClassified.shape
    good = bad = 0.0
    nbNeutral = 0.0
    for i, img in enumerate(validClassified):
        answer = numpy.argmax(img)
        label = numpy.argmax(validY[i])
        if answer == 6:
            nbNeutral += 1.0
        if answer == label:
            good += 1.0
        else:
            bad += 1.0
    print "Valid good class rate", good / validClassified.shape[0]
    print "% Neutral response", nbNeutral / validClassified.shape[0]

    print "Make submission file"
    submissionName = "{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}.txt".format(
        int(good / validClassified.shape[0] * 100),
        patchSize,
        validationSize,
        kMeansBatchSize,
        kMeansIter,
        centroidsNumber,
        logIter,
        logWeightDecay
    )
    if os.path.exists(submissionName):
        os.remove(submissionName)
    f = open(submissionName, 'wb')
    testFeatures = applyUnsupervisedTraining(testX, dataMean, dataStd, projectionMapping, centroids)
    print testFeatures.shape
    testClassified = logR.classify(testFeatures.T).T
    print testClassified.shape
    for i, img in enumerate(testClassified):
        answer = numpy.argmax(img)
        f.write("{0}\n".format(answer))

    f.close()

    print "Done"