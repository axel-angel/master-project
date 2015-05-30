#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--keep-2D', type=int, required=True)
    args = parser.parse_args()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    assert len(ls) == len(X)

    print "Save NPZ"
    ls2 = np.array([ (l >> args.keep_2D) & 1 for l in ls ])
    np.savez_compressed(args.out_npz, X, ls2)
