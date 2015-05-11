#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import numpy as np
import argparse
import utils
from random import randint
import multiprocessing
from utils import *
from collections import defaultdict
from random import Random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--num', type=int, required=True)
    parser.add_argument('--label', type=int, nargs='+', required=True)
    args = parser.parse_args()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    count = X.shape[0]

    trs = [2, 4]
    print "Transformations:\n\t%s" % ("\n\t".join(map(repr, trs)))

    def process( (x, l) ):
        xs = []
        for v in reversed(trs):
            xs.append( (l, img_shift_x(x, -v), -v) )
        xs.append( (l, x, 0) )
        for v in trs:
            xs.append( (l, img_shift_x(x, +v), +v) )
        return xs

    print "Generate images"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = []
    Xls = filter(lambda (x,l): l in args.label, np.array([ X, ls ]).T)
    itr = pool.imap_unordered(process, Xls)
    for i, xs in enumerate(itr):
        res.append(xs)
        sys.stdout.write("\rTransforming: %.0f%%" % (i * 100. / len(Xls)))
        sys.stdout.flush()
    pool.terminate()
    print ""

    rand = Random(1)
    out_imgs = []
    out_labels = []
    for i in xrange(args.num):
        xs, ys = rand.sample(res, k=2)
        # pick a similar pair
        idx = rand.randint(0, len(xs) - 2)
        (l1, i1, v1), (l2, i2, v2) = xs[idx:idx+2]
        out_imgs.append( np.array([ i1, i2 ]) )
        out_labels.append( 1 )
        # pick a disimilar pair
        (l3, i3, v3) = rand.choice(ys)
        out_imgs.append( np.array([ i1, i3 ]) )
        out_labels.append( 0 )

        sys.stdout.write("\rPairing: %.0f%%" % (i * 100. / args.num))
        sys.stdout.flush()

    print ""
    print "Write NPZ"
    np.savez_compressed(args.out_npz, out_imgs, out_labels)
