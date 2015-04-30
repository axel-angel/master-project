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
    distances = defaultdict(list)
    try:
        #num_cores = multiprocessing.cpu_count()
        #pool = multiprocessing.Pool(num_cores)
        #for i, xs in enumerate(pool.imap_unordered(process, reader)):
        for (i, image, label) in reader:
            xs = process((i, image, label))
            for (label, name, pt) in xs:
                distances[(label, name)].append(pt)
            sys.stdout.write("\rRunning: %i" % (i))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "\nStopping as requested"

    print ""
    print "Distances"
    for (label, name), ds in sorted(distances.iteritems()):
        center = reduce(operator.add, ds) / len(ds)
        ptdots = ( np.dot((d - center), (d - center)) for d in ds )
        var = np.sqrt(reduce(operator.add, ptdots))
        print "%s:%s %f" % (label, name, var)
