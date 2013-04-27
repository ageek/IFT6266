"""
A Pylearn2 Dataset object for accessing the data for the
Kaggle facial-keypoint-detection contest for the IFT 6266 H13 course.
"""
__authors__ = 'Vincent Archambault-Bouffard'
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "Vincent Archambault-Bouffard"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import os
import numpy as np

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

# The number of features in the Y vector
numberOfKeyPoints = 30


class KeypointDataset(DenseDesignMatrix):
    """
    A Pylearn2 Dataset object for accessing the data for the
    Kaggle facial-keypoint-detection contest for the IFT 6266 H13 course.
    """

    def __init__(self, which_set, keypointNumber,
                 base_path='/Users/Archi/Documents/University/IFT6266/IFT6266/Keypoints/Experiments/MLP',
                 start=None,
                 stop=None,
                 preprocessor=None,
                 fit_preprocessor=False,
                 axes=('b', 0, 1, 'c'),
                 fit_test_preprocessor=False):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                   If you are using this on the DIRO filesystem, you
                   can just use the default value. If you are using this
                   at home, you should download the .csv files from
                   Kaggle and set base_path to the directory containing
                   them.
        fit_preprocessor: True if the preprocessor is allowed to fit the
                   data.
        fit_test_preprocessor: If we construct a test set based on this
                    dataset, should it be allowed to fit the test set?
        """

        self.test_args = locals()
        self.test_args['which_set'] = 'public_test'
        self.test_args['fit_preprocessor'] = fit_test_preprocessor
        del self.test_args['start']
        del self.test_args['stop']
        del self.test_args['self']

        X, y = loadFromNumpy(which_set, keypointNumber)

        if start is not None:
            assert which_set != 'public_test'
            assert isinstance(start, float)
            assert isinstance(stop, float)
            assert start >= 0
            assert start < stop
            assert stop <= 1
            nStop = int(stop * X.shape[0])
            nStart = int(start * X.shape[0])

            X = X[nStart:nStop, :]
            if y is not None:
                y = y[nStart:nStop, :]

        view_converter = DefaultViewConverter(shape=[48, 48, 1], axes=axes)
        super(KeypointDataset, self).__init__(X=X, y=y, view_converter=view_converter)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5

    def get_test_set(self):
        return KeypointDataset(**self.test_args)


def fileNames(which_set, kpNumber):
    if which_set == "train":
        X_file = "NumpyData/keypoint_{0}_X.npy".format(kpNumber)
        Y_file = "NumpyData/keypoint_{0}_Y.npy".format(kpNumber)
    elif which_set == "public_test":
        X_file = "NumpyData/keypoint_{0}_test.npy".format(kpNumber)
        Y_file = None
    else:
        raise ValueError("Unrecognized dataset name: " + which_set)
    return X_file, Y_file


def loadFromNumpy(which_set, kpNumber):
    X_file, Y_file = fileNames(which_set, kpNumber)
    if X_file is None:
        return None, None

    path = os.path.join(X_file)
    if not os.path.exists(path):
        return None, None
    X = np.load(path)

    if Y_file is not None:
        path = os.path.join(Y_file)
        if not os.path.exists(path):
            return None, None
        Y = np.load(path)
    else:
        Y = None

    return X, Y