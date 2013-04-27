__author__ = "Vincent Archambault-Bouffard"


import sys

sys.path.append("/Users/Archi/Documents/University/pylearn2")

from pylearn2.config import yaml_parse

# Import yaml file that specifies the model to train
with open("mlp_keypoints.yaml", "r") as f:
    yamlCode = f.read()

# Training the model
train = yaml_parse.load(yamlCode)  # Creates the object from the yaml file
train.main_loop() # Starts training