__author__ = "Vincent Archambault-Bouffard"

import os
import csv
import numpy as np
import gc


# The number of features in the Y vector
numberOfKeyPoints = 30


def loadFacialKeypointDataset(which_set,
                              base_path='/Users/Archi/Documents/University/IFT6266/IFT6266/Keypoints',
                              seed=42):

    X, y = loadFromNumpy(base_path, which_set)
    if X is None or (y is None and which_set == "train"):
        print "Load from CSV"
        files = {'train': 'keypoints_train.csv', 'public_test': 'keypoints_test.csv'}

        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)

        path = os.path.join(base_path, filename)

        csv_file = open(path, 'r')

        reader = csv.reader(csv_file)

        # Discard header
        reader.next()


        if which_set == 'train':
            y = np.ones((7049, 30), dtype='float32') * -1
            X = np.zeros((7049, 96*96), dtype='float32')
        else:
            y = None
            X = np.zeros((1783, 96*96), dtype='float32')

        for i, row in enumerate(reader):
            if i % 100 == 0:
                print i
                #gc.collect()
                
            if which_set == 'train':
                y_float = readKeyPoints(row)
                X_row_str = row[numberOfKeyPoints]  # The image is at the last position
                y[i, :] = y_float
            else:
                _, X_row_str = row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: np.float32(x), X_row_strs)
            X[i,:] = X_row

        saveForNumpy(base_path, which_set, X, y)

    else:
        print "Load from .npy"


    np.random.seed(seed)
    permut = np.random.permutation(X.shape[0])
    X = X[permut, :]
    if y is not None:
        y = y[permut, :]

    return X, y


def fileNames(which_set):
    if which_set == "train":
        X_file = "keypoints_train_X.npy"
        Y_file = "keypoints_train_Y.npy"
    elif which_set == "public_test":
        X_file = "keypoints_test_X.npy"
        Y_file = None
    else:
        raise ValueError("Unrecognized dataset name: " + which_set)
    return X_file, Y_file


def loadFromNumpy(base_path, which_set):
    X_file, Y_file = fileNames(which_set)
    if X_file is None:
        return None, None

    path = os.path.join(base_path, X_file)
    if not os.path.exists(path):
        return None, None
    X = np.load(path)

    if Y_file is not None:
        path = os.path.join(base_path, Y_file)
        if not os.path.exists(path):
            return None, None
        Y = np.load(path)
    else:
        Y = None

    return X, Y


def saveForNumpy(base_path, which_set, X, Y):
    X_file, Y_file = fileNames(which_set)
    if X_file is None:
        return

    path = os.path.join(base_path, X_file)
    np.save(path, X)

    if Y_file is not None:
        path = os.path.join(base_path, Y_file)
        np.save(path, Y)


def readKeyPoints(row):
    """
    Reads the list of keypoints from a row in the csv file
    """
    kp = [-1] * numberOfKeyPoints
    for i in range(numberOfKeyPoints):
        if row[i] is not None and row[i] != "":
            kp[i] = np.float32(row[i])
    return kp