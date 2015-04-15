#!/usr/bin/python
# -*- coding: utf-8 -*-

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
              'name': '%s:%i:%i' % (tr, x, y),
              'steps': lambda x=x,y=y: range(x, y, np.sign(y-x)) }
            for k, (tr,x,y) in enumerate(args.transfo) ]
    trs_len = len(trs)
    print "Transformations: %s" % ("\n\t".join(map(repr, trs)))

    count = 0
    count_all = 0
    correct = 0
    correct_vmaxs = defaultdict(list) # extreme disto value correctly classified
    labels_set = set()

    print "Test network against transformations"
    try:
        for i, image, label in reader:
            for tr in trs:
                f, name = tr['f'], tr['name']
                for v in tr['steps']():
                    out = net.forward_all(data=np.asarray([ f(image, v) ]))
                    plabel = int(out['prob'][0].argmax(axis=0))

                    count_all += 1
                    iscorrect = label == plabel
                    if iscorrect:
                        correct += 1
                    else:
                        break

                correct_vmaxs[(label, name)].append(v)

            count += 1

            sys.stdout.write("\rRunning: %i" % (count))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "\nStopping as requested"

    print ""
    print "Extremum correct classification:"
    print "(l, tr) | avg ± stddev [count]"
    for ((l, name), vs) in sorted(correct_vmaxs.iteritems()):
        print "(%i, %s) | %f ± %f [%i]" \
                % (l, name, np.average(vs), np.std(vs), len(vs))
