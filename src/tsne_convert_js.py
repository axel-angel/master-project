#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from random import random
from math import sin, cos
from utils import partition
import json

def conv(d): # legacy conversion of label to int
    d['l'] = int(d['l'])
    return d

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-js', type=str, required=True)
    args = parser.parse_args()

    X = np.load(args.in_npz)
    Y = np.array([ X['pts'], X['infos'] ]).T

    ds = [ dict(x=pt[0], y=pt[1], **conv(i)) for pt, i in Y ]
    with open(args.out_js, 'wb') as fd:
        fd.write('var X = ')
        json.dump(ds, fd)
        fd.write(';')
