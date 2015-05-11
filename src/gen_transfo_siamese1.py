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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    args = parser.parse_args()

    print "Load dataset"
    X = np.load(args.in_npz)
    xs = X['arr_0']
    ls = X['arr_1']
    count = xs.shape[0]

    trs = [2, 4]
    print "Transformations:\n\t%s" % ("\n\t".join(map(repr, trs)))

    def process( (x, l) ):
        xs = [ (l, x, 0) ]
        for v in trs:
            xs.append( (l, img_shift_x(x, +v), +v) )
            xs.append( (l, img_shift_x(x, -v), -v) )
        return xs

    print "Generate images"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = []
    itr = pool.imap_unordered(process, np.array([ xs, ls ]).T)
    for i, x in enumerate(itr):
        res.append(x)
        sys.stdout.write("\rRunning: %i" % (i))
        sys.stdout.flush()
    pool.terminate()
    print ""

    print "Write NPZ"
    xs2_img = np.array([ x for xs2 in res for l, x in xs2 ])
    xs2_l = np.array([ l for xs2 in res for l, x in xs2 ])
    np.savez_compressed(args.out_npz, xs2_img, xs2_l)
