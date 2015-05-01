#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import numpy as np
import argparse
from collections import defaultdict
import multiprocessing
from utils import npz_reader, gen_adversial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    print "Load dataset and model"
    reader = npz_reader(args.in_npz)

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    labels = set(range(0, 10)) # FIXME: suppose MNIST
    def process( (i, img, l) ):
        mylabels = labels - set([ l ])
        return [ gen_adversial(net, img, l, tl) for tl in mylabels ]

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = defaultdict(list)
    count = 0
    try:
        for i, xss in enumerate(pool.imap_unordered(process, reader)):
            for xs in xss:
                if xs is None:
                    continue

                for k, v in xs.iteritems():
                    res[k].append(v)
                count += 1

            sys.stdout.write("Progress %i, found %i\r" % (i, count))
            sys.stdout.flush()

    except KeyboardInterrupt:
        print "\nStopping as requested"
    pool.terminate()

    print "Write NPZ"
    np.savez_compressed(args.out_npz, **res)
