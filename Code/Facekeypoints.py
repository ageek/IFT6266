__author__ = 'Vincent Archambault-Bouffard'

#The purpose of this module is to compute keypoints for all faces for the kaggle competition :
# Emotion and identity detection from face images.

import sys
#sys.path.append("/u/archambv/Documents/ubi_faces/ubi_faces/face_tracker/bin/")

import os
import numpy as np
import Image
import glob
import cv
import bin.pythonConv as pc
from contest_dataset import ContestDataset


def convertDataSetToImages(savePath=None, extension=None):
    """
    Converts the data set into images
    """
    if extension is None:
        extension = "png"

    if savePath is None:
        savePath = "/u/archambv/Documents/IFT6266/IFT6266/Images"

    if not os.path.isdir(savePath):
        os.makedirs(savePath)

    print "Loading data"
    # Build train and test datasets and Get Topological view
    datasetTrain = ContestDataset(which_set='train')
    train = datasetTrain.get_topological_view()
    train = datasetTrain.adjust_for_viewer(train)

    datasetTest = ContestDataset(which_set='public_test')
    test = datasetTest.get_topological_view()
    test = datasetTest.adjust_for_viewer(test)
    print train.shape
    print test.shape

    print "Saving as image file"
    #train, test = importAsNumpyArray(datasetPath)
    i = 1
    for row in train:
        row *= 0.5
        row += 0.5
        row *= 255
        row = np.cast['uint8'](row)
        img = Image.fromarray(row[:, :, 0])
        img2 = img.resize((48 * 2, 48 * 2), Image.ANTIALIAS)
        img2.save(os.path.join(savePath, 'train_{0}.{1}'.format(i, extension)))
        i += 1
        if i > 2:
            break
    i = 1
    for row in test:
        row *= 0.5
        row += 0.5
        row *= 255
        row = np.cast['uint8'](row)
        img = Image.fromarray(row[:, :, 0])
        img2 = img.resize((48 * 6, 48 * 6), Image.ANTIALIAS)
        img2.save(os.path.join(savePath, 'test_{0}.{1}'.format(i, extension)))
        i += 1
        if i > 2:
            break


def runFaceTracker(imagePath, extension=None):
    if extension is None:
        extension = "png"

    # Looping though all files
    for f in glob.glob(os.path.join(imagePath, "*.{0}".format(extension))):

        #loading IplImage object
        image = cv.LoadImage(f, cv.CV_LOAD_IMAGE_COLOR)

        #creating string from the image
        img = image.tostring()

        list_img = [img]

        #parameters
        nChannels = image.nChannels
        width = image.width
        height = image.height

        #calling face detect
        result = pc.face_detect(list_img, nChannels, width, height)

        #saving result to file
        f = open(os.path.join(imagePath, f.replace(".{0}".format(extension), ".txt")))
        f.write(result)
        f.close()

if __name__ == "__main__":
    convertDataSetToImages("/u/archambv/Documents/IFT6266/IFT6266/Images", "png")
    #runFaceTracker("/u/archambv/Documents/IFT6266/IFT6266/Images", "png")