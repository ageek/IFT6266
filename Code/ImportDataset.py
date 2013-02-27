__author__ = 'Vincent Archambault-Bouffard'

import numpy
import os
import Image


class ImportOptions:
    """
    Import options for the KaggleDataset methods

    validationSetPercentage : The percentage of kaggle train data to keep for validation
    validationSetMethod : How to choose the validation set
                            - Random : At random
                            - LastImages : The last images on the kaggle train set
    validationSetSeed : What seed to use in case validationSetMethod is Random.
    """

    def __init__(self):
        self.validationSetPercentage = 0.1
        self.validationSetMethod = "Random"  # Random or LastImages
        self.validationSetSeed = 5487


class KaggleDataset:
    """
    Class to load the kaggle dataset under various format and under various training/validation split
    """

    def __init__(self):
        # Insert the path to the folder containing the kaggle CSV file
        self.datasetPath = "KaggleData/"

    def loadCSV(self):
        numpy.loadtxt(os.path.join(self.datasetPath, "train.csv"), skiprows=1, dtype='float32')

    def loadAsPILimages(self, importOptions=None):
        """
        Loads the dataset as PIL images.
        Returns (train,validation,test) each set as a python list
        """
        if importOptions is None:
            importOptions = ImportOptions()

        trainImages = []
        validationImages = []
        testImages = []

        kaggleTrain = numpy.loadtxt(os.path.join(self.datasetPath, "train.csv"), skiprows=1, dtype='float32')
        kaggleTest = numpy.loadtxt(os.path.join(self.datasetPath, "test.csv"), skiprows=1, dtype='float32')

        for i, row in enumerate(kaggleTrain):
            imgArray = kaggleTrain[i].resize((48, 48))
            trainImages.append(Image.fromarray(imgArray))

        return trainImages, None, None


if __name__ == "__main__":
    kd = KaggleDataset()
    images, _, _ = kd.loadAsPILimages()
    for img in images:
        img.show()