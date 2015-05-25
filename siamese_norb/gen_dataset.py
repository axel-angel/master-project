#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from random import Random
from itertools import izip

def info_dict(ii):
    return dict(instance=ii[0], elevation=ii[1], azimuth=ii[2],
                lighting=ii[3], pair=i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-train-npz', type=str, required=True)
    parser.add_argument('--out-test-npz', type=str, required=True)
    parser.add_argument('--pair', type=int, nargs='+', default=[])
    parser.add_argument('--category', type=int, nargs='+', default=[])
    parser.add_argument('--instance', type=int, nargs='+', default=[])
    parser.add_argument('--lighting', type=int, nargs='+', default=[])
    parser.add_argument('--elevation', type=int, nargs='+', default=[])
    parser.add_argument('--azimuth', type=int, nargs='+', default=[])
    parser.add_argument('--train-count', type=float, default=660)
    parser.add_argument('--shuffle', action='store_true', default=True)
    args = parser.parse_args()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    infos = npz['arr_2']

    Xlsi = [ (x, l, info_dict(ii))
             for (xs, l, ii) in izip(X, ls, infos)
             if not args.category  or l in args.category
             if not args.instance  or ii[0] in args.instance
             if not args.elevation or ii[1] in args.elevation
             if not args.azimuth   or ii[2] in args.azimuth
             if not args.lighting  or ii[3] in args.lighting
             for i, x in enumerate(xs)
             if not args.pair or i in args.pair ]

    print "Filtered %i samples" % (len(Xlsi))

    if args.shuffle:
        Random(42).shuffle(Xlsi)

    print "Save NPZ"
    X2 = np.array([ x for (x, l, ii) in Xlsi ])
    ls2 = np.array([ l for (x, l, ii) in Xlsi ])
    infos2 = np.array([ ii for (x, l, ii) in Xlsi ])
    np.savez(args.out_train_npz, X2[:args.train_count], ls2[:args.train_count],
            infos2[:args.train_count])
    np.savez(args.out_test_npz, X2[args.train_count:], ls2[args.train_count:],
            infos2[args.train_count:])
