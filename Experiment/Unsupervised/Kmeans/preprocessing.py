__author__ = 'Vincent Archambault-Bouffard'

import numpy


def contrastNormalization(patches, stdConstant=0.001):
    """Does the canonical preprocessing. Patches are row-wise

    DC Centering
    Contrast Normalization"""

    patches -= patches.mean(1)[:, None]   # DC CENTERING
    patches /= patches.std(1)[:, None] + stdConstant  # CONTRAST NORMALIZATION
    return patches


def standardScore(patches, stdConstant=0.001):
    """Computes the standard scroe. Patches are row-wise

    Mean Centering
    Variance Normalization"""

    dataMean = patches.mean(0)  # Data Mean centering
    dataStd = patches.std(0) + stdConstant  # Data Normalization

    patches -= dataMean[None, :]  # Data Mean centering
    patches /= dataStd[None, :]  # Data Normalization

    patches = apply_standardScore(patches, dataMean, dataStd)
    return patches, dataMean, dataStd


def apply_standardScore(patches, dataMean, dataStd):
    """Applies the canonical preprocessing. Patches are row-wise"""

    patches -= dataMean[None, :]  # Data Mean centering
    patches /= dataStd[None, :]  # Data Normalization

    return patches


def pca_whitening(data, var_fraction, whiteningConstant=0.01):
    """ principal components analys retaining as many components as required to
        retain var_fraction of the variance

    Returns projected data, projection mapping, inverse mapping"""
    from numpy.linalg import eigh

    u, v = eigh(numpy.cov(data, rowvar=0))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum() <= u.sum() * var_fraction]
    u += whiteningConstant
    numprincomps = u.shape[0]
    V = ((u ** (-0.5))[:numprincomps][None, :] * v[:, :numprincomps]).T
    W = (u ** 0.5)[:numprincomps][None, :] * v[:, :numprincomps]
    return numpy.dot(data, V.T), V, W


def apply_pca_whitening(X, projectionMapping):
    """
    Applies the PCA whitening projection mapping to the data
    """
    return numpy.dot(X, projectionMapping.T)