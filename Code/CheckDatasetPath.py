__author__ = 'Vincent Archambault-Bouffard'

import sys
import os


def checkDatasetPath():
    """
    Check if the root folder of this project is in the python path
    Check if the contest dataset is in the python path.
    Check also for Xavier Bouthillier
    """
    if not any("IFT6266" in s for s in sys.path):
        # We insert the path
        d = os.path.join(os.getenv("HOME"),  "Documents/IFT6266/IFT6266")
        if os.path.isdir(d):
            sys.path.append(d)
        else:
            d = os.path.join(os.getenv("HOME"),  "Documents/University/IFT6266/")
            if os.path.isdir(d):
                sys.path.append(d)

    if not any("ContestDataset" in s for s in sys.path):
        # We insert the path
        d = os.path.join(os.getenv("HOME"),  "Documents/University/IFT6266/ContestDataset")
        sys.path.append(d)

    if not any("ift6266kaggle/transform/" in s for s in sys.path):
        # We insert the path
        d = os.path.join(os.getenv("HOME"),  "Documents/IFT6266/ift6266kaggle/transform")
        if os.path.isdir(d):
            sys.path.append(d)
        else:
            d = os.path.join(os.getenv("HOME"),  "Documents/University/IFT6266/transform")
            if os.path.isdir(d):
                sys.path.append(d)