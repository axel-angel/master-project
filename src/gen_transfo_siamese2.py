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
    args = parser.parse_args()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    nss = npz['nss']
    assert len(ls) == len(X)
    assert len(nss) == len(X)
    count = X.shape[0]
    neighs = nss.shape[1]

    trs = [3, 6]
    print "Transformations:\n\t%s" % ("\n\t".join(map(repr, trs)))

    def process( (x, l, i) ):
        xs = []
        for v in reversed(trs):
            xs.append( (l, img_shift_x(x, -v), -v) )
        xs.append( (l, x, 0) )
        for v in trs:
            xs.append( (l, img_shift_x(x, +v), +v) )
        return (i, xs)

    print "Generate images"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = {}
    Xls = np.array([ X, ls, range(count) ]).T
    itr = pool.imap_unordered(process, Xls)
    for it, (i, xs) in enumerate(itr):
        res[i] = xs
        sys.stderr.write("\rTransforming: %.0f%%" % (it * 100. / len(Xls)))
    pool.terminate()
    print ""

    rand = Random(1)
    out_imgs = []
    out_labels = []
    for i in xrange(count):
        xs = res[i]
        ns = nss[i]
        # pick similar pairs
        for idx in xrange(0, len(xs) - 1):
            # pick the sample translations
            (l1, i1, v1), (l2, i2, v2) = xs[idx:idx+2]
            out_imgs.append( np.array([ i1, i2 ]) )
            out_labels.append( 1 )
            # pick translated neighbors as well
            for n in ns:
                (l3, i3, v3) = res[n][idx]
                out_imgs.append( np.array([ i1, i3 ]) )
                out_labels.append( 1 )
        # pick disimilar pairs
        js = rand.sample([ k for k in xrange(count) if k not in ns ], k=neighs)
        for j in js:
            ys = res[j]
            for idx in xrange(0, len(ys) - 1):
                (l1, i1, v1) = xs[idx]
                (l4, i4, v4) = ys[idx]
                out_imgs.append( np.array([ i1, i4 ]) )
                out_labels.append( 0 )

        sys.stderr.write("\rPairing: %.0f%%" % (i * 100. / count))

    print ""
    print "Write NPZ"
    np.savez_compressed(args.out_npz, out_imgs, out_labels)
