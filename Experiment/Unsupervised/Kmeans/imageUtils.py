__author__ = ['Roland Memisevic', 'Vincent Archambault-Bouffard']

import pylab
import numpy


def display_color(M, border=0, bordercolor=None, *imshow_args, **imshow_keyargs):
    """ Display an array of rgb images.

    M is the matrix in (data,row,column,color) indexing

    The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
    """
    if not bordercolor:
        bordercolor = [0.0, 0.0, 0.0]
    bordercolor = numpy.array(bordercolor)[None, None, :]

    numimages = len(M)
    M = M.copy()

    # Transform the image to make sure they have uniform scaling
    for i in range(M.shape[0]):
        M[i] -= M[i].flatten().min()
        M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3

    # Compute the grid of the final image
    n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    im = numpy.array(bordercolor) * numpy.ones(
        ((height + border) * n1 + border, (width + border) * n0 + border, 1), dtype='<f8')

    # Add each image to the grid
    for i in range(n0):
        for j in range(n1):
            if i * n1 + j < numimages:
                # Compute position in the grid
                vStart = j * (height + border) + border
                vEnd = (j + 1) * (height + border) + border
                hStart = i * (width + border) + border
                hEnd = (i + 1) * (width + border) + border
                # Add border to the right
                imageWithBorder = numpy.concatenate(
                    (M[i * n1 + j, :, :, :],
                     bordercolor * numpy.ones((height, border, 3), dtype=float)),
                    axis=1)
                # Add border below
                imageWithBorder = numpy.concatenate(
                    (imageWithBorder,
                     bordercolor * numpy.ones((border, width + border, 3), dtype=float)),
                    axis=0)
                im[vStart:vEnd, hStart:hEnd, :] = imageWithBorder
    imshow_keyargs["interpolation"] = "nearest"
    pylab.imshow(im, *imshow_args, **imshow_keyargs)
    pylab.show()