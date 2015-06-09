#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import caffe
import argparse
import multiprocessing
from itertools import izip, imap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--layer', type=str, default='loss')
    args = parser.parse_args()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']

    def process( (x, l) ):
        x_caffe = np.array([ x ]) / 255. # normalize
        l_caffe = np.array([[[[ l ]]]])
        res = net.forward(pair_data=x_caffe, sim=l_caffe)
        return res[args.layer]

    print "Forward"
    losses = []
    try:
        #num_cores = multiprocessing.cpu_count()
        #pool = multiprocessing.Pool(num_cores)
        #itr = pool.imap_unordered(process, izip(X, ls))
        itr = imap(process, izip(X, ls))
        for it, res in enumerate(itr):
            losses.append( res )
            sys.stderr.write("\rRunning: %i" % (it))
    except KeyboardInterrupt:
        print "\nStopping as requested"
    finally:
        #pool.terminate()
        pass
    print ""

    print "Loss: %f ± %f" % (np.average(losses), np.std(losses))
