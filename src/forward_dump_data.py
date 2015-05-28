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
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-data', type=str, required=True)
    args = parser.parse_args()

    print "Load network"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Load data"
    X = np.load(args.in_npz)
    xs = X['arr_0']
    ls = X['arr_1']
    infos = X['arr_2']

    print "Forward output"
    dims = xs.shape[-2:]
    res = net.forward_all(data=xs.reshape(-1, 1, *dims), blobs=[args.layer])
    out = {}
    for k in res:
        out[k] = flat_shape(res[k])

    print "Save into GNUplot data"
    with open(args.out_data, 'w') as fd:
        for ii, xs in izip(infos, out[args.layer]):
            cols = map(lambda x: "%f" % (x), xs) + [ str(ii['elevation']) ]
            fd.write( (" ".join(cols)) + "\n")