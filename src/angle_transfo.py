#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import caffe
import numpy as np
import argparse
from collections import defaultdict
from utils import npz_reader, parse_transfo, parse_transfo_random
from random import randint
import utils
import multiprocessing
import operator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--transfo-random', type=parse_transfo, nargs='+',
            action='append', default=None)
    parser.add_argument('--npz', type=str, default=None)
    parser.add_argument('--layer', type=str, default='ip1')
    args = parser.parse_args()

    print "Load model"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    reader = npz_reader(args.npz)
    trs = utils.parse_transfo_random(args.transfo_random)

    trs_len = len(trs)
    print "Transformations: %s" % ("\n\t".join(map(repr, trs)))

    def process( (i, image, label) ):
        image1_caffe = image.reshape(1, *image.shape)
        data = np.asarray([ image1_caffe ])
        out = net.forward_all(data=data, blobs=[ args.layer ])
        pt1 = utils.flat_shape(out[args.layer][0])

        diffs = []
        dists = []
        for tr in trs:
            f, name = tr['f'], tr['name']

            image2 = f(image)
            image2_caffe = image2.reshape(1, *image.shape)
            data = np.asarray([ image2_caffe ])
            out = net.forward_all(data=data, blobs=[ args.layer ])
            pt2 = utils.flat_shape(out[args.layer][0])

            # compute angle diff and distance
            norm = np.linalg.norm(pt1) * np.linalg.norm(pt2)
            diff = pt2 - pt1
            dist = np.linalg.norm(diff)

            diffs.append( (label, name, diff) )
            dists.append( (label, name, dist) )

        return (diffs, dists)

    print "Start parallel computation"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    diffs_all = defaultdict(list)
    dists_all = defaultdict(list)
    try:
        itr = pool.imap_unordered(process, reader)
        for i, (diffs, dists) in enumerate(itr):
            for (label, name, diff) in diffs:
                diffs_all[(label, name)].append(diff)
            for (label, name, dist) in dists:
                dists_all[(label, name)].append(dist)
            sys.stdout.write("\rRunning: %i" % (i))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "\nStopping as requested"
    pool.terminate()

    print ""
    print "Mean angle"
    diffs_vars = []
    for (label, name), diffs in sorted(diffs_all.iteritems()):
        arr = np.array(diffs)
        mean = np.mean(arr, axis=0)
        mean_norm = np.linalg.norm(mean)
        angle_diffs = [ np.arccos( np.dot(mean, pt)
                            / (mean_norm * np.linalg.norm(pt)) )
                        for pt in diffs ]
        angle_mean = np.mean(angle_diffs)
        angle_var = np.var(angle_diffs)
        print "%s %s %.5f %.5f" % (label, name, angle_mean, angle_var)
        diffs_vars.append(angle_var)
    print "avg %.5f" % (np.mean(diffs_vars))

    print "Distance norm"
    dists_vars = []
    for (label, name), dists in sorted(dists_all.iteritems()):
        arr = np.array(dists)
        mean = np.mean(arr, axis=0)
        var = np.var(arr, axis=0)
        print "%s %s %.0f %.0f" % (label, name, mean, var)
        dists_vars.append(var)
    print "avg %.5f" % (np.mean(dists_vars))
