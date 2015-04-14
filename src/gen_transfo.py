#!/usr/bin/python

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
    parser.add_argument('--transfo', type=parse_transfo, nargs='+',
            action='append', default=[])
    args = parser.parse_args()

    print "Load dataset"
    X = np.load(args.in_npz)
    xs = X['arr_0']
    ls = X['arr_1']
    count = xs.shape[0]

    trs = []
    def fold_transfo(f, (tr, x, y)):
        trf = getattr(utils, 'img_%s' % (tr))
        return lambda i: trf(f(i), randint(x, y))
    for transfos in args.transfo:
        foldedf = reduce(fold_transfo, transfos, lambda i: i)
        trs.append(foldedf)

    print "Transformations:\n\t%s" % ("\n\t".join(map(repr, args.transfo)))

    def process((i, (x, l))):
        xs2 = [ (l, f(x)) for f in trs ]

        if i%1000 == 0:
            sys.stdout.write("Progress %4.1f%% (%i/%i)\r" \
                    % (100.*i/count, i, count))
            sys.stdout.flush()

        return xs2

    print "Generate images"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = pool.map(process, enumerate(np.array([ xs, ls ]).T))
    print ""

    print "Write NPZ"
    xs2_img = np.array([ x for xs2 in res for l, x in xs2 ])
    xs2_l = np.array([ l for xs2 in res for l, x in xs2 ])
    np.savez_compressed(args.out_npz, xs2_img, xs2_l)
