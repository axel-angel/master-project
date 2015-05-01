#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import caffe
import numpy as np
import argparse
from collections import defaultdict
from utils import lmdb_reader, npz_reader, parse_transfo
import utils
import multiprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--transfo-grid', type=parse_transfo, nargs='+',
            action='append', default=None)
    group1.add_argument('--transfo-random', type=parse_transfo, nargs='+',
            action='append', default=None)
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--lmdb', type=str, default=None)
    group2.add_argument('--npz', type=str, default=None)
    args = parser.parse_args()

    print "Load model"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()
    print "args", vars(args)
    if args.lmdb != None:
        reader = lmdb_reader(args.lmdb)
    if args.npz != None:
        reader = npz_reader(args.npz)

    if args.transfo_grid:
        trs = utils.parse_transfo_grid(args.transfo_grid)
    if args.transfo_random:
        trs = utils.parse_transfo_random(args.transfo_random)

    trs_len = len(trs)
    print "Transformations: %s" % ("\n\t".join(map(repr, trs)))

    count = 0
    accuracies = defaultdict(list) # extreme disto value correctly classified
    labels_set = set()

    print "Test network against transformations"
    def process( (i, image, label) ):
        res = []
        for tr in trs:
            f, name = tr['f'], tr['name']

            image2 = f(image)
            image2_caffe = image2.reshape(1, *image.shape)
            out = net.forward_all(data=np.asarray([ image2_caffe ]))
            plabel = int(out['prob'][0].argmax(axis=0))

            iscorrect = label == plabel
            res.append((label, name, int(iscorrect)))

        return res

    print "Start parallel testing"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = []
    try:
        for i, xs in enumerate(pool.imap_unordered(process, reader)):
            for (label, name, iscorrect) in xs:
                accuracies[(label, name)].append(iscorrect)
            sys.stdout.write("\rRunning: %i" % (i))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "\nStopping as requested"
    pool.terminate()

    print ""
    print "Extremum correct classification:"
    print "(l, tr) | accu ± stddev [count]"
    count = 0
    total_avg = []
    for ((l, name), vs) in sorted(accuracies.iteritems()):
        avg = np.average(vs)
        total_avg.append(avg)
        count += len(vs)
        print "%i:%s  %f  %f  %i" % (l, name, avg, np.std(vs), len(vs))
    # average
    print "%s:%s  %f  %f  %i" \
            % ('avg', 'ALL', np.average(total_avg), np.std(total_avg), count)
