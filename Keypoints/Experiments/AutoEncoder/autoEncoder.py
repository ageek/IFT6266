from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from theano import function
import numpy as np
import loadData
from PIL import Image, ImageDraw


rootPath = '/Users/Archi/Documents/University/IFT6266/IFT6266/Keypoints'

kpList = ['left_eye_center_x',
                    'left_eye_center_y',
                    'right_eye_center_x',
                    'right_eye_center_y',
                    'left_eye_inner_corner_x',
                    'left_eye_inner_corner_y',
                    'left_eye_outer_corner_x',
                    'left_eye_outer_corner_y',
                    'right_eye_inner_corner_x',
                    'right_eye_inner_corner_y',
                    'right_eye_outer_corner_x',
                    'right_eye_outer_corner_y',
                    'left_eyebrow_inner_end_x',
                    'left_eyebrow_inner_end_y',
                    'left_eyebrow_outer_end_x',
                    'left_eyebrow_outer_end_y',
                    'right_eyebrow_inner_end_x',
                    'right_eyebrow_inner_end_y',
                    'right_eyebrow_outer_end_x',
                    'right_eyebrow_outer_end_y',
                    'nose_tip_x',
                    'nose_tip_y',
                    'mouth_left_corner_x',
                    'mouth_left_corner_y',
                    'mouth_right_corner_x',
                    'mouth_right_corner_y',
                    'mouth_center_top_lip_x',
                    'mouth_center_top_lip_y',
                    'mouth_center_bottom_lip_x',
                    'mouth_center_bottom_lip_y']
mapping = dict(zip(kpList, range(30)))

indexOfAll = [mapping['left_eye_center_x'], mapping['left_eye_center_y'],
              mapping['right_eye_center_x'], mapping['right_eye_center_y'],
              mapping['nose_tip_x'], mapping['nose_tip_y'],
              mapping['mouth_center_bottom_lip_x'], mapping['mouth_center_bottom_lip_y']]


def runExperiment():
    # Import yaml file that specifies the model to train
    with open("model.yaml", "r") as f:
        yamlCode = f.read()

    # Training the model
    train = yaml_parse.load(yamlCode)  # Creates the object from the yaml file
    train.main_loop()  # Starts training


def testModel():
    print "Load pkl file"
    model = serial.load("autoEncoder.pkl")

    # Get the validation error
    #monitor = model.monitor
    #channels = monitor.channels
    #validationError = channels["valid_objective"].val_record[-1]

    # Compute test
    print "Load dataset"
    dataset = yaml_parse.load(model.dataset_yaml_src)
    dataset = dataset.get_test_set()

    print "Preparing"
    # use smallish batches to avoid running out of memory
    batch_size = 100
    model.set_batch_size(batch_size)
    # dataset must be multiple of batch size of some batches will have
    # different sizes. theano convolution requires a hard-coded batch size
    m = dataset.X.shape[0]
    extra = batch_size - m % batch_size
    assert (m + extra) % batch_size == 0
    if extra > 0:
        dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
                                                        dtype=dataset.X.dtype)), axis=0)
    assert dataset.X.shape[0] % batch_size == 0

    X = model.get_input_space().make_batch_theano()
    Y = model.reconstruct(X)
    f = function([X], Y)

    y = []

    print "Reconstruct"
    for imgIdx in xrange(dataset.X.shape[0] / batch_size):
        if imgIdx % 100 == 0:
            print imgIdx
        x_arg = dataset.X[imgIdx * batch_size:(imgIdx + 1) * batch_size, :]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))

    y = np.concatenate(y)
    assert y.shape[0] == dataset.X.shape[0]
    # discard any zero-padding that was used to give the batches uniform size
    return y[:m], dataset.X[:m]


def sampleImage(x, y):
    yFinal = None
    for i in np.random.randint(0, 100, 6):
        image = x[i].reshape(96, 96, 1)
        image = np.cast['uint8'](image)
        image = Image.fromarray(image[:, :, 0])
        image.save("autoEncoder_{}_X.png".format(i))

        image = y[i].reshape(96, 96, 1)
        image = np.cast['uint8'](image)
        image = Image.fromarray(image[:, :, 0])
        yFinal = image.copy()
        image.save("autoEncoder_{}_Y.png".format(i))

    return yFinal


def drawKeypointsOnImage(img, keyPoints):
    """
    Returns an RGB image with the keypoints added to it.
    Green for left side and red for right side. (relative to subject)
    Original author = Pierre-Luc Carrier
    """

    cp = img.copy().convert("RGB")

    draw = ImageDraw.Draw(cp)
    draw.setink("#00ff00")

    leftFill = (0, 255, 0)
    rightFill = (255, 0, 0)

    left_eye_center_x = 0
    left_eye_inner_corner_x = 4
    left_eye_outer_corner_x = 6
    left_eyebrow_inner_end_x = 12
    left_eyebrow_outer_end_x = 14
    mouth_left_corner_x = 22

    for i in range(len(keyPoints) / 2):
        if keyPoints[i * 2] is not None and keyPoints[i * 2 + 1] is not None:
            if i * 2 in [left_eye_center_x,
                         left_eye_inner_corner_x,
                         left_eye_outer_corner_x,
                         left_eyebrow_inner_end_x,
                         left_eyebrow_outer_end_x,
                         mouth_left_corner_x,
                         left_eye_center_x]:
                fill = leftFill
            else:
                fill = rightFill
            draw.ellipse((int(keyPoints[i * 2]), int(keyPoints[i * 2 + 1]),
                          int(keyPoints[i * 2]) + 4, int(keyPoints[i * 2 + 1]) + 4),
                         fill=fill)

    del draw
    return cp


def splitFullSparse(X, Y):
    indexSparse = Y[:, 10] == -1
    xFull = X[np.logical_not(indexSparse), :]
    yFull = Y[np.logical_not(indexSparse), :]
    xSparse = X[indexSparse, :]
    ySparse = Y[indexSparse, :]
    return xFull, yFull, xSparse, ySparse


def getAverage():
    print "Building keypoint dataset"
    X, Y = loadData.loadFacialKeypointDataset("train", base_path=rootPath)
    Xtest, _ = loadData.loadFacialKeypointDataset("public_test", base_path=rootPath)
    xFull, yFull, xSparse, ySparse = splitFullSparse(X, Y)

    Yaverage = np.average(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    yFullaverage = np.average(yFull, axis=0)
    yFullstd = np.std(yFull, axis=0)

    avg = Yaverage
    std = Ystd
    for i in range(30):
        if i not in indexOfAll:
            avg[i] = yFullaverage[i]
            std[i] = yFullstd[i]

    return avg


if __name__ == "__main__":
    Y, X = testModel()
    print Y.shape
    print X.shape
    print "Sample"
    yFinal = sampleImage(X, Y)
    yFinal.save("yFinal.png")
    print "Average"
    avg = getAverage()
    print "yFinal"
    img = drawKeypointsOnImage(yFinal, avg)
    img.save("reconstructKeypoints.png")