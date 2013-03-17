__author__ = 'Vincent Archambault-Bouffard'

import sys
from pylearn2.config import yaml_parse
sys.path.append('/Users/Archi/Documents/University/IFT6266/')
sys.path.append('/Users/Archi/Documents/University/IFT6266/IFT6266')
sys.path.append('/Users/Archi/Documents/University/IFT6266/IFT6266/Code')
from Code.CheckDatasetPath import checkDatasetPath

# Make sure we have the path to the dataset
checkDatasetPath()
sys.path.append('/Users/Archi/Documents/University/IFT6266/ift6266kaggle/transform')

# Import yaml file that specifies the model to train
<<<<<<< HEAD
with open("convNet_2layer_32kernel_0.5Momemtum.yaml", "r") as f:
=======
with open("convNet_standardization.yaml", "r") as f:
>>>>>>> New experiment
    yamlCode = f.read()

# Training the model
train = yaml_parse.load(yamlCode)  # Creates the object from the yaml file
train.main_loop() # Starts training