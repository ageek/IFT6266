# coding=utf-8
__author__ = "Vincent Archambault-Bouffard"

import loadData
import numpy as np
import csv
import os
from PIL import Image, ImageDraw
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from theano import function

rootPath = '/data/lisatmp/ift6266h13/ContestDataset/'
#rootPath = '/Users/Archi/Documents/University/IFT6266/IFT6266/Keypoints'

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

indexOfSomeMissing = [x for x in range(30) if x not in indexOfAll]

span = 24


def splitFullSparse(X, Y):
    indexSparse = Y[:, 10] == -1
    xFull = X[np.logical_not(indexSparse), :]
    yFull = Y[np.logical_not(indexSparse), :]
    xSparse = X[indexSparse, :]
    ySparse = Y[indexSparse, :]
    return xFull, yFull, xSparse, ySparse


def makeSubmission(y, out_path):
    submission = []
    with open(rootPath+'/submissionFileFormat.csv', 'rb') as cvsTemplate:
        reader = csv.reader(cvsTemplate)
        for row in reader:
            submission.append(row)

    for row in submission[1:]:
        imgIdx = int(row[1]) - 1
        keypointName = row[2]
        keyPointIndex = mapping[keypointName]
        row.append(y[imgIdx, keyPointIndex])

    if os.path.exists(out_path):
        os.remove(out_path)

    with open(out_path, 'w') as cvsTemplate:
        writer = csv.writer(cvsTemplate)
        for row in submission:
            writer.writerow(row)


def extract_region(X, kpNumber, avg):
    # Crops a region of 48 * 48 centered on the keypoints if possible
    avgX = int(avg[2*kpNumber])
    avgY = int(avg[2*kpNumber+1])

    # Correction to keep the subimage inside the original image
    def computeOffset(center, span):
        if center - span < 0:
            return - (center - span)
        elif center + span > 96:
            return 96 - (center + span)
        return 0

    offsetX = computeOffset(avgX, span)
    offsetY = computeOffset(avgY, span)

    xx = X.reshape(X.shape[0], 96, 96)
    xx = xx[:, avgY - span + offsetY:avgY + span + offsetY, avgX - span + offsetX:avgX + span + offsetX]
    xx = xx.reshape(xx.shape[0], 4 * span * span)
    return xx, offsetX, offsetY


# Function to generate the dataset avec le numéro du keypoint en .npy file
# Generic keypoints dataset

# YAML generic avec le numéro du keypoint
# Boucle pour chaque keypoint ... train model
# Boucle pour chaque keypoint loading model et faire prédiction
# Make submission !!

def loadAndTrainObject(folderName):
    # Import yaml file that specifies the model to train
    with open("{0}/model.yaml".format(folderName), "r") as f:
        yamlCode = f.read()

    yamlCode = yamlCode.replace("{1}", folderName)
    for i in range(15):
        modelCode = yamlCode.replace("{0}", str(i))

        # Training the model
        print "Training model {0}".format(i)
        model = yaml_parse.load(modelCode)  # Creates the object from the yaml file
        model.main_loop()

    return


def buildKeyPointsDataset():
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

    a = """
        print "Sample image"
        for i in range(15):
            image = X[4].reshape(96, 96, 1)
            image = np.cast['uint8'](image)
            image = Image.fromarray(image[:, :, 0])
            kp = [0] * 30
            kp[2*i] = avg[2*i]
            kp[2*i+1] = avg[2*i+1]
            image = drawKeypointsOnImage(image, kp)
            image.save("sample_keypoint_{0}.png".format(kpList[2*i]))

        return
        """
    # Save to file
    if not os.path.exists("NumpyData/"):
        os.mkdir("NumpyData")
    np.save("NumpyData/averageKp.npy", avg)
    np.save("NumpyData/stdKp.npy", std)

    for i in range(15):
        print "Dataset #{0}".format(i)
        if 2*i in indexOfAll:
            xx = X
            yy = Y[:, 2*i:2*i+2]
        else:
            xx = xFull
            yy = yFull[:, 2*i:2*i+2]

        # Crop only the good region
        xx, _, _ = extract_region(xx, i, avg)
        xxTest, _, _ = extract_region(Xtest, i, avg)
        print xx.shape
        print yy.shape

        np.save("NumpyData/keypoint_{0}_X.npy".format(i), xx)
        np.save("NumpyData/keypoint_{0}_Y.npy".format(i), yy)
        np.save("NumpyData/keypoint_{0}_test.npy".format(i), xxTest)

        print "Sample image"
        if not os.path.exists("SampleKeypoint/"):
            os.mkdir("SampleKeypoint")

        for h in np.random.randint(0, xx.shape[0], 6):
            image = xx[h].reshape(2*span, 2*span, 1)
            image = np.cast['uint8'](image)
            image = Image.fromarray(image[:, :, 0])
            image.save("SampleKeypoint/keypoint_{0}_sample_{1}.png".format(kpList[2*i], h))


def testModel(model_path):
    model = serial.load(model_path)

    # Get the validation error
    monitor = model.monitor
    channels = monitor.channels
    validationError = channels["valid_objective"].val_record[-1]

    # Compute test
    dataset = yaml_parse.load(model.dataset_yaml_src)
    dataset = dataset.get_test_set()

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
    Y = model.fprop(X)
    f = function([X], Y)

    y = []

    for imgIdx in xrange(dataset.X.shape[0] / batch_size):
        x_arg = dataset.X[imgIdx * batch_size:(imgIdx + 1) * batch_size, :]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))

    y = np.concatenate(y)
    assert y.shape[0] == dataset.X.shape[0]
    # discard any zero-padding that was used to give the batches uniform size
    return y[:m], validationError



def buildingY(folderName):
    avg = np.load("NumpyData/averageKp.npy")
    std = np.load("NumpyData/stdKp.npy")
    Xtest, _ = loadData.loadFacialKeypointDataset("public_test", base_path=rootPath)
    y = np.zeros((Xtest.shape[0], 30))
    valAvg = 0.0
    valCount = 0.0
    for i in range(15):
        print "Test model {0}".format(i)
        y[:, 2 * i:2 * i + 2], valErr = testModel("{1}/generic_keypoint_{0}.pkl".format(i, folderName))
        valAvg += (3 if 2*i in indexOfAll else 1) * valErr
        valCount += 3 if 2*i in indexOfAll else 1
    valAvg /= valCount
    with open('{0}/valid_{1}.txt'.format(folderName, int(valAvg)), "w") as f:
        f.write(str(valAvg))
    return Xtest, avg, std, y


def limitingY(avg, std, y):
    nbChange = 0
    for idx, line in enumerate(y):
        for i in range(30):
            if line[i] - avg[i] >= 3 * std[i]:
                y[idx, i] = avg[i] + 1.5 * std[i]
                #print "Changing {0}th keypoint for {1}. {2}, {3}, {4}".format(i, idx, line[i], avg[i], std[i])
                nbChange += 1
            if line[i] - avg[i] <= - 3 * std[i]:
                y[idx, i] = avg[i] - 1.5 * std[i]
                #print "Changing {0}th keypoint for {1}. {2}, {3}, {4}".format(i, idx, line[i], avg[i], std[i])
                nbChange += 1
    print "Changed {0} over {1}".format(nbChange, y.shape[0] * 30)


def sampleImage(Xtest, folderName, y):
    for i in np.random.randint(0, Xtest.shape[0], 6):
        image = Xtest[i].reshape(96, 96, 1)
        image = np.cast['uint8'](image)
        image = Image.fromarray(image[:, :, 0])
        # Superpose key points
        image = drawKeypointsOnImage(image, y[i])

        image.save("{0}/individualmlp_{1}.png".format(folderName, i))


def runExperiment(folderName):
    print "Loading model"
    loadAndTrainObject(folderName)

    print "Building Y"
    Xtest, avg, std, y = buildingY(folderName)

    print "Limiting y values"
    limitingY(avg, std, y)

    print "Make submission"
    makeSubmission(y, "{0}/individual_mlp_submission.csv".format(folderName))

    # Transform to PIL image
    print "Sample image"
    sampleImage(Xtest, folderName, y)

if __name__ == "__main__":
    #buildKeyPointsDataset()
    runExperiment("ConvRectifiedLinear")
    #runExperiment("testPipeline")