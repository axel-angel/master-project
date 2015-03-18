#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from utils import partition

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--filter', type=str, required=True)
    parser.add_argument('--top', type=int, default=1)
    parser.add_argument('--transfo', type=str, nargs='*', default=[])
    parser.add_argument('--value', type=int, nargs='*', default=[])
    args = parser.parse_args()

    # in-npz: [array([-0.36007209, -3.17370899]), {'src': '9', 'tr': 'identity', 'l': -1, 'v': 0}]], dtype=object) 

    print "Load data"
    npz = np.load(args.in_npz)
    X = np.array([ npz['pts'], npz['infos'] ]).T

    Xn, Xd = partition(lambda y: y[1]['src'] == "dataset", X)
    Y2 = filter(lambda y: args.filter in y[1]['src'], Xn)

    for pt, i in Y2:
        if len(args.transfo) > 0 and i['tr'] not in args.transfo:
            continue
        if len(args.value) > 0 and i['v'] not in args.value:
            continue

        print "Closest to %s: %s" % (pt, i)
        xs = sorted(Xd, key=lambda y: np.sum((y[0] - pt) ** 2))[:args.top]
        for pt2, i2 in xs:
            print " - %s: %s" % (pt2, i2)
