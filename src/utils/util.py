import os
import random
from argparse import Namespace
from pathlib import Path
from pprint import pprint
from typing import Union

import numpy as np


def set_random_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)


def print_argparse_arguments(p: Namespace, bar: int = 50) -> None:

    """
    from https://github.com/ka10ryu1/jpegcomp
    Visualize argparse arguments.
    Arguments:
        p : parse_arge() object.
        bar : the number of bar on the output.
    """
    print("PARAMETER SETTING")
    print("-" * bar)
    args = [(i, getattr(p, i)) for i in dir(p) if "_" not in i[0]]
    for i, j in args:
        if isinstance(j, list):
            print("{0}[{1}]:".format(i, len(j)))
            for k in j:
                print("\t{}".format(k))
        else:
            print("{0:25}:{1}".format(i, j))
    print("-" * bar)
