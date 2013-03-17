__author__ = "Vincent Archambault-Bouffard"

import csv
import os
import numpy

rootPath = '/Users/Archi/Documents/University/IFT6266/ContestDataset'


def loadDataset():
    """
    Returns the images from train.csv and test.csv and the labels
    """
    trainX = []
    trainY = []
    testX = []

    # For train.csv
    path = os.path.join(rootPath, 'train.csv')
    csv_file = open(path, 'r')
    reader = csv.reader(csv_file)

    # Discard header
    reader.next()

    for row in reader:
        y_str, x_str = row
        y = numpy.zeros(7).reshape(1, 7)
        y[0, int(y_str)] = 1
        trainY.append(y)

        x_str = x_str.split(' ')
        x_str = map(lambda x: float(x), x_str)
        trainX.append(numpy.asarray(x_str, dtype=float).reshape(1, 48, 48))

    csv_file.close()

    # For test.csv
    path = os.path.join(rootPath, 'test.csv')
    csv_file = open(path, 'r')
    reader = csv.reader(csv_file)

    # Discard header
    reader.next()

    for row in reader:
        x_str = row[0].split(' ')
        x_str = map(lambda x: float(x), x_str)
        testX.append(numpy.asarray(x_str, dtype=float).reshape(1, 48, 48))

    csv_file.close()

    return numpy.concatenate(trainX), numpy.concatenate(trainY), numpy.concatenate(testX)

if __name__ == "__main__":
    print "Loading Dataset"
    trainX, trainY, testX = loadDataset()

    print trainY.shape, trainY.sum(), trainY.mean(axis=0)
    print trainX.shape, trainX.min(), trainX.max()
    print testX.shape, testX.min(), testX.max()
    import imageUtils
    imageUtils.display_greyscale(trainX[0:9])
    imageUtils.display_greyscale(testX[0:9])