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
from itertools import izip, imap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    Xls = np.array([ X, ls ]).T
    count = X.shape[0]

    def process( (i, (x, l)) ):
        distsq = np.sum((x - X)**2, axis=(1,2))
        ids = sorted(enumerate(distsq), key=lambda (j,d): d)
        ns = np.array([j for (j,d) in ids if j != i]) # skip itself
        return (i, ns[0:args.count])

    print "Compute distances"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    itr = pool.imap_unordered(process, enumerate(Xls))
    nss = np.zeros((count, args.count))
    for it, (i, ns) in enumerate(itr):
        nss[i,:] = ns
        sys.stderr.write("\rComputing: %.0f%%" % (it * 100. / count))
    pool.terminate()
    print ""

    print "Write NPZ"
    np.savez_compressed(args.out_npz, nss=nss)
