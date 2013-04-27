# coding=utf-8
__author__ = "Vincent Archambault-Bouffard"

"""Always answer the average"""

import loadData
import numpy as np
import csv
import os
from PIL import Image, ImageDraw
from sklearn import linear_model


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

mapping = dict(zip(['left_eye_center_x',
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
                    'mouth_center_bottom_lip_y'], range(30)))

indexOfAll = [mapping['left_eye_center_x'], mapping['left_eye_center_y'],
              mapping['right_eye_center_x'], mapping['right_eye_center_y'],
              mapping['nose_tip_x'], mapping['nose_tip_y'],
              mapping['mouth_center_bottom_lip_x'], mapping['mouth_center_bottom_lip_y']]

indexOfSomeMissing = [x for x in range(30) if x not in indexOfAll]


def splitFullSparse(X, Y):
    indexSparse = Y[:, 10] == -1
    xFull = X[np.logical_not(indexSparse), :]
    yFull = Y[np.logical_not(indexSparse), :]
    xSparse = X[indexSparse, :]
    ySparse = Y[indexSparse, :]
    return xFull, yFull, xSparse, ySparse


def makeSubmission(y, out_path):
    submission = []
    with open('/Users/Archi/Documents/University/IFT6266/IFT6266/Keypoints/submissionFileFormat.csv', 'rb') as cvsTemplate:
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

if __name__ == "__main__":
    print "allAverage started"
    X, Y = loadData.loadFacialKeypointDataset("train")
    print X.shape, Y.shape
    xFull, yFull, xSparse, ySparse = splitFullSparse(X, Y)
    print xFull.shape, yFull.shape
    print xSparse.shape, ySparse.shape

    print "Computing and substracting average"
    Yaverage = np.average(Y, axis=0)
    yFullaverage = np.average(yFull, axis=0)

    avg = Yaverage
    for i in range(30):
        if i not in indexOfAll:
            avg[i] = yFullaverage[i]

    # Linear regression avec yFull et keypoints donc certains sont manquants
    print "Training when missing"
    yFull -= avg
    yFull = yFull[:, indexOfSomeMissing]
    regrWhenMissing = linear_model.RidgeCV(alphas=[5, 1, 0.5, 0.1, 0.01], normalize=True)
    regrWhenMissing.fit(xFull, yFull)

    # Linear regression avec toutes les données pour les keypoints complets
    print "Training when All"
    Y -= avg
    Y = Y[:, indexOfAll]
    print Y.shape
    print X.shape
    regrWhenAll = linear_model.RidgeCV(alphas=[5, 1, 0.5, 0.1, 0.01], normalize=True)
    regrWhenAll.fit(X, Y)

    print "Building Y"
    Xtest, _ = loadData.loadFacialKeypointDataset("public_test")
    y = np.zeros((Xtest.shape[0], 30))
    for z, data in enumerate(Xtest):
        yWhenMissing = regrWhenMissing.predict(data)
        yWhenAll = regrWhenAll.predict(data)
        for i in range(30):
            if i in indexOfAll:
                pos = indexOfAll.index(i)
                ans = yWhenAll[pos]
            else:
                pos = indexOfSomeMissing.index(i)
                ans = yWhenMissing[pos]
            y[z, i] = ans + avg[i]

    print "Make submission"
    makeSubmission(y, "linreg_submission.csv")

    # Transform to PIL image
    print "Sample image"
    for i in np.random.randint(0, Xtest.shape[0], 6):
        image = Xtest[i].reshape(96, 96, 1)  # Take first image as exemple
        image = np.cast['uint8'](image)
        image = Image.fromarray(image[:, :, 0])
        # Superpose key points
        image = drawKeypointsOnImage(image, y[i])

        image.save("linearRegression_{0}.png".format(i))