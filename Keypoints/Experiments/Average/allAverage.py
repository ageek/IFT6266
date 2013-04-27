__author__ = "Vincent Archambault-Bouffard"

"""Always answer the average"""

import loadData
import numpy as np
import csv
import os
from PIL import Image, ImageDraw


def sampleImages(X, avg, std):
    # Transform to PIL image
    for i in np.random.randint(0, 7000, 6):
        image = X[i].reshape(96, 96, 1)  # Take first image as exemple
        image = np.cast['uint8'](image)
        image = Image.fromarray(image[:, :, 0])
        # Superpose key points
        image = drawKeypointsOnImage(image, avg, std)

        image.save("averageKeypoints_{0}.png".format(i))


def drawKeypointsOnImage(img, keyPoints, std):
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
            draw.ellipse((int(keyPoints[i * 2]) - std[i * 2]/2.0, int(keyPoints[i * 2 + 1]) - std[i * 2 + 1]/2.0,
                          int(keyPoints[i * 2]) + std[i * 2]/2.0, int(keyPoints[i * 2 + 1]) + std[i * 2 + 1]/2.0),
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
        keypointName = row[2]
        keyPointIndex = mapping[keypointName]
        row.append(y[keyPointIndex])

    if os.path.exists(out_path):
        os.remove(out_path)

    with open(out_path, 'w') as cvsTemplate:
        writer = csv.writer(cvsTemplate)
        for row in submission:
            writer.writerow(row)


def printReport(avg, std, report):
    print "Name, average, std, 1std, 2std, 3std"
    for idx in range(30):
        line = ""
        line += "{0}, {1:.2f}, {2:.2f}, ".format(kpList[idx], avg[idx], std[idx])
        for i in range(1, 4):
            line += "{0:.2f}%, ".format(report[idx, i] * 100)
        print line


if __name__ == "__main__":
    print "allAverage started"
    X, Y = loadData.loadFacialKeypointDataset("train")
    print X.shape, Y.shape
    xFull, yFull, xSparse, ySparse = splitFullSparse(X, Y)
    print xFull.shape, yFull.shape
    print xSparse.shape, ySparse.shape

    Yaverage = np.average(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    yFullaverage = np.average(yFull, axis=0)
    yFullstd = np.std(yFull, axis=0)

    print Yaverage.shape
    print yFullaverage.shape

    nbTotal = X.shape[0]
    nbFull = xFull.shape[0]
    avg = Yaverage
    std = Ystd
    for i in range(30):
        if i not in indexOfAll:
            avg[i] = yFullaverage[i]
            std[i] = yFullstd[i]

    # Count number of example in Std
    report = np.zeros((30, 6))
    for i in range(1, 4):
        for idx in range(30):
            if idx not in indexOfAll:
                total = nbFull
                xx = yFull
            else:
                total = nbTotal
                xx = Y
            count = np.sum((np.abs(xx[:, idx] - avg[idx]) - (1.0 * i * std[idx]) <= 0).astype(int))
            maX = np.max((np.abs(xx[:, idx] - avg[idx])))
            report[idx, i] = count * 1.0 / total
            report[idx, 4] = maX

    printReport(avg, std, report)

    #makeSubmission(avg, "average_submission.csv")
    #sampleImages(X, avg, std)


