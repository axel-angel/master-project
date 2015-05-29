#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
from utils import *
from itertools import izip

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--norb-label', type=str, default=None)
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-data', type=str, required=True)
    args = parser.parse_args()

    print "Load network"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Load data"
    npz = np.load(args.in_npz)
    xs = npz['arr_0']
    if args.norb_label:
        ls = [ ii[args.norb_label] for ii in npz['arr_2'] ]
    else:
        ls = npz['arr_1']

    print "Forward output"
    dims = xs.shape[-2:]
    res = net.forward_all(data=xs.reshape(-1, 1, *dims), blobs=[args.layer])
    out = {}
    for k in res:
        out[k] = flat_shape(res[k])

    print "Save into GNUplot data"
    with open(args.out_data, 'w') as fd:
        for l, xs in izip(ls, out[args.layer]):
            cols = map(lambda x: "%f" % (x), xs) + [ str(l) ]
            fd.write( (" ".join(cols)) + "\n")
