__author__ = 'Vincent Archambault-Bouffard'

import os
HOME = os.environ['HOME']

import numpy
import numpy.random
import pylab
from dispims_color import dispims_color
import online_kmeans

patchsize = 6
numhid = 100
rng = numpy.random.RandomState(1)


def crop_patches_color(image, keypoints, patchSize):
    patches = numpy.zeros((len(keypoints), 3*patchSize**2))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0]-patchSize/2:k[0]+patchSize/2, k[1]-patchSize/2:k[1]+patchSize/2,:].flatten()
    return patches


def pca(data, var_fraction):
    """ principal components, retaining as many components as required to
        retain var_fraction of the variance

    Returns projected data, projection mapping, inverse mapping, mean"""
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=1, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*var_fraction]
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
    W = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
    return numpy.dot(V,data), V, W


#CROP PATCHES
print "generating patches"
trainims = numpy.loadtxt("cifar_mini_images_train.txt")
patches = numpy.concatenate([crop_patches_color(im.reshape(3, 32, 32).transpose(1,2,0), numpy.array([rng.randint(patchsize, 32-patchsize, 100), rng.randint(patchsize, 32-patchsize, 100)]).T, patchsize) for im in trainims]).astype("float32")
R = rng.permutation(patches.shape[0])
patches = patches[R, :]
print "done"


#WHITEN PATCHES
print "whitening"
meanstd = patches.std()
patches -= patches.mean(1)[:,None]
patches /= patches.std(1)[:,None] + 0.1 * meanstd
patches -= patches.mean(0)[None,:]
patches /= patches.std(0)[None,:]
pcadata, pca_backward, pca_forward = pca(patches.T, .9)
featurelearndata = pcadata.T.astype("float32")
del pcadata
numpatches = featurelearndata.shape[0]
print "numpatches: ", numpatches
print "done"



#TRAIN KMEANS
f1 = pylab.figure()
Rinit = rng.permutation(numhid)
W = featurelearndata[Rinit]
print "training kmeans"
for epoch in range(20):
    W = online_kmeans.kmeans(featurelearndata, numhid, Winit=W, numepochs=1, learningrate=0.01*0.8**epoch)
    W_ = numpy.dot(pca_forward,W.T).T.reshape(numhid, patchsize, patchsize, 3)
    dispims_color(W_)
    pylab.draw(); pylab.show()

print "done"

f2 = pylab.figure()
dispims_color(pca_backward.reshape(pca_backward.shape[0], 6, 6, 3))




