#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
from utils import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=str, nargs='+')
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    args = parser.parse_args()

    print "Load network"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Load data"
    npz = np.load(args.in_npz)
    xs = npz['arr_0']
    ls = npz['arr_1']

    print "Forward output"
    dims = xs.shape[-2:]
    res = net.forward_all(data=xs.reshape(-1, 1, *dims), blobs=args.layer)
    out = {}
    for k in res:
        out[k] = flat_shape(res[k])

    np.savez_compressed(args.out_npz, labels=ls, **out)
