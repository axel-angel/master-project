#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from random import random
from math import sin, cos
from utils import partition
import json
from collections import defaultdict

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
    Y = np.array([ X['pts'], X['infos'], range(X['pts'].shape[0]) ]).T

    src_set = defaultdict(int)
    label_set = defaultdict(int)
    tr_set = defaultdict(int)
    for _, info, _ in Y:
        src_set[info['src']] += 1
        label_set[str(info['l'])] += 1
        tr_set[info['tr']] += 1

    ds = [ dict(x=pt[0], y=pt[1], i=i, **conv(info)) for pt, info, i in Y ]
    with open(args.out_js, 'wb') as fd:
        fd.write('var X = ')
        json.dump(ds, fd)
        fd.write(';')

        for v in ['src_set', 'label_set', 'tr_set']:
            fd.write('var %s = ' % (v))
            print locals()[v]
            json.dump(locals()[v], fd)
            fd.write(';')
