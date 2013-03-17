__author__ = ['Roland Memisevic', 'Vincent Archambault-Bouffard']

import numpy


def crop_patches_color(image, patchsize, keypoints=None):
    """
    Crop patches of the specified size around the keypoints

    keypoints can be :
     iterable( (x,y), (x,y) ... ) -> List of the center of the patches
     interger -> Number of patches to crop at random position
     None -> 10 patches crop at random position

    Returns the patches in a matrix with data row wise
    """

    if not hasattr(keypoints, '__iter__'):
        if keypoints is None:
            keypoints = 10
        keypoints = zip(numpy.random.randint(patchsize // 2, image.shape[0] - patchsize // 2, keypoints),
                        numpy.random.randint(patchsize // 2, image.shape[1] - patchsize // 2, keypoints))

    patches = numpy.zeros((len(keypoints), patchsize, patchsize, 3))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0] - patchsize / 2:k[0] + patchsize / 2,
                              k[1] - patchsize / 2:k[1] + patchsize / 2,
                              :]
    return patches

def crop_patches_grayscale(image, patchsize, keypoints=None):
    """
    Crop patches of the specified size around the keypoints

    keypoints can be :
     iterable( (x,y), (x,y) ... ) -> List of the center of the patches
     interger -> Number of patches to crop at random position
     None -> 10 patches crop at random position

    Returns the patches in a matrix with data row wise
    """
    # ToDo fix bug when patchsize is odd
    patchHalfSize = patchsize // 2
    if not hasattr(keypoints, '__iter__'):
        if keypoints is None:
            keypoints = 10
        keypoints = zip(numpy.random.randint(patchHalfSize, image.shape[0] - patchHalfSize, keypoints),
                        numpy.random.randint(patchHalfSize, image.shape[1] - patchHalfSize, keypoints))

    patches = numpy.zeros((len(keypoints), patchsize, patchsize))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0] - patchHalfSize:k[0] + patchHalfSize,
                              k[1] - patchHalfSize:k[1] + patchHalfSize]
    return patches