#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import caffe
import numpy as np
import lmdb
import argparse
from collections import defaultdict
from utils import lmdb_reader, npz_reader, parse_transfo
import utils

def mkCombinaisons(ranges):
    vals = map(lambda x: [x], ranges[0])
    for r in ranges[1:]:
        print "r", r
        ys = []
        vals2 = [ xs + [y] for xs in vals for y in r ]
        vals = vals2
    return vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--transfo', type=parse_transfo, nargs='+',
            action='append', default=[], required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lmdb', type=str, default=None)
    group.add_argument('--npz', type=str, default=None)
    args = parser.parse_args()

    print "Load model"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()
    print "args", vars(args)
    if args.lmdb != None:
        reader = lmdb_reader(args.lmdb)
    if args.npz != None:
        reader = npz_reader(args.npz)

    trs = []
    for k, transfos in enumerate(args.transfo):
        name = "+".join("("+ ('%s:%+i:%+i:%i' % (tr, x, y, dt)) + ")"
                        for (tr, x, y, dt) in transfos)
        ranges = [ range(x, y, dt*np.sign(y-x)) for (tr,x,y,dt) in transfos ]
        values = mkCombinaisons(ranges)
        for vs in values:
            myf = lambda i: i
            def reducer( f, (tf, v) ):
                return lambda i: tf(f(i), v)
            fs = [ getattr(utils, 'img_%s' % (tr)) for (tr,x,y,dt) in transfos ]
            f = reduce(reducer, zip(fs, vs), myf)
            trs.append({ 'f': f, 'name': name })

    trs_len = len(trs)
    print "Transformations: %s" % ("\n\t".join(map(repr, trs)))

    count = 0
    accuracies = defaultdict(list) # extreme disto value correctly classified
    labels_set = set()

    print "Test network against transformations"
    try:
        for i, image, label in reader:
            for tr in trs:
                f, name = tr['f'], tr['name']

                image2 = f(image)
                image2_caffe = image2.reshape(1, *image.shape)
                out = net.forward_all(data=np.asarray([ image2_caffe ]))
                plabel = int(out['prob'][0].argmax(axis=0))

                iscorrect = label == plabel
                accuracies[(label, name)].append(int(iscorrect))

            count += 1

            sys.stdout.write("\rRunning: %i" % (count))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "\nStopping as requested"

    print ""
    print "Extremum correct classification:"
    print "(l, tr) | accu ± stddev [count]"
    total_avg = []
    for ((l, name), vs) in sorted(accuracies.iteritems()):
        avg = np.average(vs)
        total_avg.append(avg)
        print "(%i, %s) | %f ± %f [%i]" % (l, name, avg, np.std(vs), len(vs))
    # average
    print "(%s, %s) | %f ± %f [%i]" \
            % ('-', 'avg', np.average(total_avg), np.std(total_avg), count)
