#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import json
from collections import defaultdict
from utils import *
import multiprocessing
from itertools import izip, chain, imap

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=str, default='ip1')
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-js', type=str, required=True)
    parser.add_argument('--label', type=int, nargs='+', default=[])
    parser.add_argument('--no-transfo', action='store_true', default=False)
    parser.add_argument('--axis1', type=int, default=0)
    parser.add_argument('--axis2', type=int, default=1)
    args = parser.parse_args()

    print "Load model"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Load data"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    Xls = np.array([ X, ls ]).T
    if len(args.label) > 0:
        Xls = filter(lambda (x, l): l in args.label, Xls)
    X = None; ls = None
    print "Filtered %i images" % (len(Xls))

    # stats per src, label, transformations
    src_set = dict(dataset=len(Xls))
    tr_set = dict(identity=len(Xls))
    label_set = defaultdict(int)
    for _, l in Xls:
        label_set[str(l)] += 1

    if args.no_transfo:
        values = [0]
    else:
        values = [0, 3, 6, -3, -6]
    print "Transformations:", values
    samples = len(Xls) * len(values)
    def process( (i, (img, l)) ):
        xs = []
        for v in values:
            img2 = img_shift_x(img, v)
            img_caffe = img2.reshape(1, *img2.shape)
            data = np.asarray([ img_caffe/255. ]) # normalize!
            out = net.forward_all(data=data, blobs=[ args.layer ])
            pt = flat_shape(out[args.layer][0])
            img64 = js_img_encode(img2) # encode64 for JS
            xs.append( (i, map(np.asscalar, pt), int(l), v, img64) )
        return xs

    # forward in parallel
    print "Processing images"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    itr = pool.imap_unordered(process, enumerate(Xls))
    info = dict(tr='shift_x', src='dataset')
    ds = []
    imgs = []
    ax1 = args.axis1
    ax2 = args.axis2
    for i, (j, pt, l, v, img64) in enumerate(chain.from_iterable(itr)):
        ds.append(dict(x=pt[ax1], y=pt[ax2], i=i, l=l, v=v, sample=j, **info))
        imgs.append(img64)
        sys.stderr.write("\rForwarding: %.0f%%" % (i * 100. / samples))
    pool.terminate()
    print ""

    # convert to json-serialisable
    print "Converting into JS"
    with open(args.out_js, 'wb') as fd:
        fd.write('var X = ')
        json.dump(ds, fd)
        fd.write(';')

        fd.write('var imgs = ')
        json.dump(imgs, fd)
        fd.write(';')

        for v in ['src_set', 'label_set', 'tr_set']:
            fd.write('var %s = ' % (v))
            json.dump(locals()[v], fd)
            fd.write(';')
