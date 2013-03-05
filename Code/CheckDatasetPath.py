__author__ = 'Vincent Archambault-Bouffard'

import sys
import os


def checkDatasetPath():
    """
    Check if the contest dataset is in the python path.
    """
    if any("ContestDataset" in s for s in sys.path):
        return

    # We insert the path for Vincent personal computer
    sys.path.append(os.path.join( os.getenv("HOME"),  "Documents/University/IFT6266/ContestDataset"))