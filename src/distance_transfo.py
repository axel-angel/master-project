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
        res = []
        for tr in trs:
            f, name = tr['f'], tr['name']

            image2 = f(image)
            image2_caffe = image2.reshape(1, *image.shape)
            data = np.asarray([ image2_caffe ])
            out = net.forward_all(data=data, blobs=[ args.layer ])
            pt = utils.flat_shape(out[args.layer][0])

            res.append((label, name, pt))

        return res

    print "Start parallel computation"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    distances = defaultdict(list)
    try:
        #from itertools import imap
        #for i, xs in enumerate(imap(process, reader)):
        for i, xs in enumerate(pool.imap_unordered(process, reader)):
            for (label, name, pt) in xs:
                distances[(label, name)].append(pt)
            sys.stdout.write("\rRunning: %i" % (i))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "\nStopping as requested"
    pool.terminate()

    print ""
    print "Distances std"
    centers = {}
    for (label, name), ds in sorted(distances.iteritems()):
        dsarr = np.array(ds)
        centers[(label, name)] = dsarr.sum(axis=0) / len(dsarr)

    print "Cross-Distances std"
    for (label, name1), ds in sorted(distances.iteritems()):
        dsarr = np.array(ds)
        center = centers[(label, name1)]
        std = np.sqrt(np.sum(np.dot(x, x) for x in dsarr - center) / len(ds))
        print "%s:%s %.1f" % (label, name, std)

        for (label2, name2), _ in sorted(distances.iteritems()):
            if label2 != label or name1 == name2:
                continue
            center2 = centers[(label2, name2)]
            std = np.sqrt(np.sum(np.dot(x, x) for x in dsarr - center2)
                    / len(ds))
            print "%s:%s->%s %.1f" % (label, name1, name2, std)
