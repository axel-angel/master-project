#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import numpy as np
import lmdb
import utils

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-lmdb', type=str, required=True)
    args = parser.parse_args()

    print "Load dataset"
    X = np.load(args.in_npz)
    xs = X['arr_0']
    if len(xs.shape) == 2: xs = xs.reshape(-1, 1, *xs.shape[-2:])
    ls = X['arr_1']
    count = xs.shape[0]

    Y = np.array([ xs, ls ]).T
    assert(count < 1000000000)

    print "Write LMDB"
    lmdb_env = lmdb.open(args.out_lmdb, map_size=1e12)
    with lmdb_env.begin(write=True) as lmdb_txn:
        for i, (x, l) in enumerate(Y):
            datum = caffe.io.array_to_datum(x, label=int(l))
            lmdb_txn.put("%010d" % (i), datum.SerializeToString())

            if i%1000 == 0:
                sys.stderr.write("Progress %4.0f%% (%i/%i)\r" \
                        % (100.*i/count, i, count))
    print ""
