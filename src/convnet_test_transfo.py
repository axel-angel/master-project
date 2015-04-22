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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--transfo', type=parse_transfo, action='append',
            default=[])
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

    trs = [ { 'f': getattr(utils, 'img_%s' % (tr)),
              'name': '%s:%+i:%+i:%i' % (tr, x, y, dt),
              'steps': lambda x=x,y=y,dt=dt: range(x, y, dt*np.sign(y-x)) }
            for k, (tr,x,y,dt) in enumerate(args.transfo) ]
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
                i_correct = 0
                i_count = 0
                for v in tr['steps']():
                    image2 = f(image, v)
                    image2_caffe = image2.reshape(1, *image.shape)
                    out = net.forward_all(data=np.asarray([ image2_caffe ]))
                    plabel = int(out['prob'][0].argmax(axis=0))

                    i_count += 1
                    iscorrect = label == plabel
                    i_correct += 1 if iscorrect else 0

                accuracies[(label, name)].append(i_correct / i_count)

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
