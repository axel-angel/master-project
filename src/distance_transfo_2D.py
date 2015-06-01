#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import caffe
import numpy as np
import argparse
from collections import defaultdict
from random import randint
from utils import *
import multiprocessing
from numpy.linalg import norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--transfo-name', type=str, required=True)
    parser.add_argument('--transfo-values', type=int, nargs='+')
    parser.add_argument('--axis-label', type=int, required=True)
    parser.add_argument('--axis-transfo', type=int, required=True)
    parser.add_argument('--npz', type=str, default=None)
    parser.add_argument('--layer', type=str, default='ip1')
    args = parser.parse_args()

    print "Load model"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    reader = npz_reader(args.npz)

    tr_f = globals().get('img_%s' % (args.transfo_name))
    trs_all = args.transfo_values
    print "Transformations: %s, [%s]" % (tr_f, ", ".join(map(repr, trs_all)))

    def process( (i, image, label) ):
        res = []
        for v in trs_all:
            image2 = tr_f(image, v)
            image2_caffe = image2.reshape(1, *image.shape)
            data = np.asarray([ image2_caffe/255. ]) # normalize!
            out = net.forward_all(data=data, blobs=[ args.layer ])
            pt = flat_shape(out[args.layer][0])

            res.append((pt, v))

        return (i, label, res)

    print "Start parallel computation"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    dss_label = defaultdict(list)
    dss_transfo = defaultdict(list)
    center_label = defaultdict(list)
    center_transfo = defaultdict(list)
    label_set = set()
    try:
        #from itertools import imap
        #for it, (i, l, res) in enumerate(imap(process, reader)):
        for it, (i, l, res) in enumerate(pool.imap_unordered(process, reader)):
            label_set.add(l)
            # measure distance between same label between distortions
            ds_transfo = [ pt[args.axis_label] for (pt,v) in res ]
            dss_transfo[l].append( np.std(ds_transfo) )
            # measure predictability along transfo axis
            res2 = zip(res[0:], res[1:])
            ds_label = [ norm(pt2[args.axis_transfo] - pt1[args.axis_transfo])
                    for (pt1,v1),(pt2,v2) in res2 ]
            dss_label[l].append( np.std(ds_label) )
            # measure centers
            for (pt,v) in res:
                center_transfo[v].append( pt[args.axis_transfo] )
                center_label[l].append( pt[args.axis_label] )
            sys.stderr.write("\rRunning: %i" % (it))
    except KeyboardInterrupt:
        print "\nStopping as requested"
    finally:
        pool.terminate()
    print ""

    label_set = sorted(label_set)
    print "\nIntra-cluster distances"
    for l in label_set:
        std = np.std( center_label[l] )
        print "l:%s %f" % (l, std)
    for v in trs_all:
        std = np.std( center_transfo[v] )
        print "t:%+i %f" % (v, std)

    print "\nInter-cluster distances"
    for (l1, l2) in zip(label_set[0:], label_set[1:]):
        d = norm( np.average(center_label[l2]) - np.average(center_label[l1]) )
        print "l:%s->%s %f" % (l1, l2, d)
    for (v1, v2) in zip(trs_all[0:], trs_all[1:]):
        d = norm( np.average(center_transfo[v2]) - np.average(center_transfo[v1]) )
        print "t:%+i->%+i %f" % (v1, v2, d)

    print "\nInter-sample and predictability"
    dss_avg_label = []
    dss_avg_transfo = []
    for l in label_set:
        ds_label = np.average( dss_label[l] )
        ds_transfo = np.average( dss_transfo[l] )
        print "%s %f %f" % (l, ds_label, ds_transfo)
        dss_avg_label.append( ds_label )
        dss_avg_transfo.append( ds_transfo )

    print "\nFinal score"
    ds_label = np.average( dss_avg_label )
    ds_transfo = np.average( dss_avg_transfo )
    print "avgs %f %f" % (ds_label, ds_transfo)
