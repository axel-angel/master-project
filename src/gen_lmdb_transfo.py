#!/usr/bin/python

import sys
import caffe
import numpy as np
import lmdb
import argparse
import utils
from random import randint
import multiprocessing

def parse_transfo(s):
    try:
        xs = s.split(':', 2)
        fmt = [str, int, int]
        return map(lambda (f,x): f(x), zip(fmt, xs))
    except:
        raise argparse.ArgumentTypeError("Invalid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-lmdb', type=str, required=True)
    parser.add_argument('--transfo', type=parse_transfo, action='append',
            default=[])
    args = parser.parse_args()

    print "Load dataset"
    X = np.load(args.in_npz)
    xs = X['arr_0']
    ls = X['arr_1']
    count = xs.shape[0]

    trs = [ { 'f': getattr(utils, 'img_%s' % (tr)), 'steps': lambda: [randint(x,y)] }
            for (tr,x,y) in args.transfo ]
    trs.append({ 'f': lambda i,v: i, 'steps': lambda: [0] })
    trs_len = len(trs)
    print "Transformations: %s" % ("\n\t".join(map(repr, trs)))

    print "Generate images"
    def process((i, (x, l))):
        trf = [ (t['f'], v) for t in trs for v in t['steps']() ]
        xs2 = []
        for j, (f, v) in enumerate(trf):
            x2 = f(x, v).reshape((1,) + x.shape)
            xs2.append((l, x2))

        if i%1000 == 0:
            sys.stdout.write("Progress %4.1f%% (%i/%i)\r" \
                    % (100.*i/count, i, count))
            sys.stdout.flush()

        return xs2

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = pool.map(process, enumerate(np.array([ xs, ls ]).T))
    print ""

    print "Write LMDB"
    lmdb_env = lmdb.open(args.out_lmdb, map_size=1e12)
    with lmdb_env.begin(write=True) as lmdb_txn:
        for i2, (l, x2) in enumerate((l, x2) for xs2 in res for l, x2 in xs2):
            datum = caffe.io.array_to_datum(x2, label=int(l))
            lmdb_txn.put("%010d" % (i2), datum.SerializeToString())

    print "Close database"
    lmdb_env.close()
