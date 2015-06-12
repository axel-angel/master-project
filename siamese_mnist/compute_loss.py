#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import caffe
import argparse
import multiprocessing
from itertools import izip, imap
from random import Random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--layer', type=str, default='loss')
    parser.add_argument('--quiet', action='store_true', default=False)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true', default=False)
    args = parser.parse_args()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    # net surgery (ensure loss uses first 2D)
    if net.params['feat'][0].data.shape[0] >= 3:
        if not args.quiet:
            print "Set 0 to 3D params"
        net.params['feat'][0].data[2:,:] = 0
        net.params['feat'][1].data[2:] = 0
        net.params['feat_p'][0].data[2:,:] = 0
        net.params['feat_p'][1].data[2:] = 0

    if not args.quiet:
        print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    assert len(X) == len(ls)

    if args.shuffle:
        if not args.quiet:
            print "Shuffle"
        idxs = np.random.RandomState(42).permutation(len(X))
        X = X[idxs]
        ls = ls[idxs]

    if args.limit != None:
        if not args.quiet:
            print "Cut to limit"
        X = X[:args.limit]
        ls = ls[:args.limit]

    def process( (x, l) ):
        x_caffe = np.array([ x ]) / 255. # normalize
        l_caffe = np.array([[[[ l ]]]])
        res = net.forward(pair_data=x_caffe, sim=l_caffe, blobs=[args.layer, 'feat', 'feat_p'])
        assert np.all( net.blobs['feat'].data[2:] == 0 )
        assert np.all( net.blobs['feat_p'].data[2:] == 0 )
        return float(res[args.layer])

    if not args.quiet:
        print "Forward"
    count = len(X)
    losses = []
    sum = 0.0
    sumsq = 0.0
    try:
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
        itr = pool.imap_unordered(process, izip(X, ls))
        #itr = imap(process, izip(X, ls))
        for it, loss in enumerate(itr):
            losses.append( loss )
            sum += loss
            sumsq += loss**2
            if not args.quiet and it > 0:
                mean = sum / (1 + it)
                std = np.sqrt( sumsq / (1 + it) - mean * mean )
                sys.stderr.write("\rLoss: %f %f [Running: %4.1f%%]" \
                        % (mean, std, it*100./count))
    except KeyboardInterrupt:
        print "\nStopping as requested"
    finally:
        #pool.terminate()
        pass
    if not args.quiet:
        print ""

    mean = sum / (1 + it)
    std = np.sqrt( sumsq / (1 + it) - mean * mean )
    print "Loss: %f %f" % (mean, std)
