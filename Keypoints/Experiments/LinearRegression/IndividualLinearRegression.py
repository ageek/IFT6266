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


if __name__ == "__main__":
    print "Individual Linear Regression started"
    X, Y = loadData.loadFacialKeypointDataset("train")
    print X.shape, Y.shape
    xFull, yFull, xSparse, ySparse = splitFullSparse(X, Y)
    print xFull.shape, yFull.shape
    print xSparse.shape, ySparse.shape

    print "Computing average and std"
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

    # Une régression linéaire par keypoints
    regressions = [None] * 15
    alpha = "CrossValid"
    for i in range(15):
        print "Training linear regression #{0}".format(i)
        if 2*i in indexOfAll:
            xx = X
            yy = Y[:, 2*i:2*i+2]
        else:
            xx = xFull
            yy = yFull[:, 2*i:2*i+2]

        # Crop only the good region
        xx, _, _ = extract_region(xx, i, avg)

        regrWhenMissing = linear_model.RidgeCV(alphas=[10, 1, 0.1], normalize=True, fit_intercept=True)
        regrWhenMissing.fit(xx, yy)
        regressions[i] = regrWhenMissing
        print "Regression {0} alpha = {1}".format(i, regrWhenMissing.alpha_)

    print "Building Y"
    Xtest, _ = loadData.loadFacialKeypointDataset("public_test")
    y = np.zeros((Xtest.shape[0], 30))
    for z, data in enumerate(Xtest):
        for i in range(15):
            xx, offsetX, offsetY = extract_region(data.reshape(1, 96, 96), i, avg)
            xx = xx.reshape(xx.shape[1])

            answer = regressions[i].predict(xx)
            y[z, 2*i] = answer[0]
            y[z, 2*i+1] = answer[1]

    print "Limiting y values"
    nbChange = 0
    for idx, line in enumerate(y):
        for i in range(30):
            if line[i] - avg[i] >= 2 * std[i]:
                y[idx, i] = avg[i] + 1.5 * std[i]
                #print "Changing {0}th keypoint for {1}. {2}, {3}, {4}".format(i, idx, line[i], avg[i], std[i])
                nbChange += 1
            if line[i] - avg[i] <= - 2 * std[i]:
                y[idx, i] = avg[i] - 1.5 * std[i]
                #print "Changing {0}th keypoint for {1}. {2}, {3}, {4}".format(i, idx, line[i], avg[i], std[i])
                nbChange += 1
    print "Changed {0} over {1}".format(nbChange, y.shape[0]*30)

    print "Make submission"
    makeSubmission(y, "individual_linreg_submission_alpha{0}.csv".format(alpha))

        # Transform to PIL image
    print "Sample image"
    for i in np.random.randint(0, Xtest.shape[0], 6):
        image = Xtest[i].reshape(96, 96, 1)
        image = np.cast['uint8'](image)
        image = Image.fromarray(image[:, :, 0])
        # Superpose key points
        image = drawKeypointsOnImage(image, y[i])

        image.save("individuallinearRegression_{0}.png".format(i))