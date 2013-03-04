__author__ = 'Vincent Archambault-Bouffard'

from pylearn2.config import yaml_parse

# Import yaml file that specifies the model to train
with open("test_mlp.yaml", "r") as f:
    yamlCode = f.read()

# Training the model
train = yaml_parse.load(yamlCode)  # Creates the Train object from the yaml file
train.main_loop() # Starts training