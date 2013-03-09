__author__ = 'Vincent Archambault-Bouffard'

from pylearn2.config import yaml_parse
from Code.CheckDatasetPath import checkDatasetPath

# Make sure we have the path to the dataset
checkDatasetPath()

# Import yaml file that specifies the model to train
with open("convNet_2layer_32kernel_0.5Momemtum.yaml", "r") as f:
    yamlCode = f.read()

# Training the model
train = yaml_parse.load(yamlCode)  # Creates the object from the yaml file
train.main_loop() # Starts training