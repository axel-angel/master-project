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
    parser.add_argument('--label', type=int, nargs='+', default=[])
    parser.add_argument('--no-transfo', action='store_true', default=False)
    parser.add_argument('--transfo-values', type=int, nargs='*', default=None)
    parser.add_argument('--transfo-name', type=str, default=None)
    parser.add_argument('--axis1', type=int, default=0)
    parser.add_argument('--axis2', type=int, default=1)
    parser.add_argument('--norb-label', type=str, default=None)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--out-js', type=str, default=None)
    g.add_argument('--out-data', type=str, default=None)
    g.add_argument('--out-npz', type=str, default=None)
    args = parser.parse_args()

    print "Load model"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Load data"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    if args.norb_label:
        ls = [ ii[args.norb_label] for ii in npz['arr_2'] ]
    else:
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

    if args.transfo_name and args.transfo_values:
        tr_f = globals().get('img_%s' % (args.transfo_name))
        values = args.transfo_values
        #values = [0, 3, 6, -3, -6]
    else:
        tr_f = img_identity
        values = [0]
    print "Transformations: %s, [%s]" % (tr_f, ", ".join(map(repr, values)))
    samples = len(Xls) * len(values)
    def process( (i, (img, l)) ):
        xs = []
        for v in values:
            img2 = tr_f(img, v)
            img_caffe = img2.reshape(1, *img2.shape)
            data = np.asarray([ img_caffe/255. ]) # normalize!
            out = net.forward_all(data=data, blobs=[ args.layer ])
            pt = flat_shape(out[args.layer][0]).reshape(-1)
            img64 = js_img_encode(img2) # encode64 for JS
            xs.append( (i, map(np.asscalar, pt), int(l), v, img64) )
        return xs

    # forward in parallel
    print "Processing images"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    itr = pool.imap_unordered(process, enumerate(Xls))
    info = dict(tr=args.transfo_name, src='dataset')
    ds = []
    imgs = []
    for i, (j, pt, l, v, img64) in enumerate(chain.from_iterable(itr)):
        ds.append(dict(i=i, l=l, v=v, sample=j, pt=pt, **info))
        imgs.append(img64)
        sys.stderr.write("\rForwarding: %.0f%%" % (i * 100. / samples))
    pool.terminate()
    print ""

    if args.out_js:
        # convert to json-serialisable
        print "Converting into JS"
        ax1 = args.axis1
        ax2 = args.axis2
        ds2 = [ dict(x=d['pt'][ax1], y=d['pt'][ax2], **d) for d in ds ]
        with open(args.out_js, 'wb') as fd:
            fd.write('var X = ')
            json.dump(ds2, fd)
            fd.write(';')

            fd.write('var imgs = ')
            json.dump(imgs, fd)
            fd.write(';')

            for v in ['src_set', 'label_set', 'tr_set']:
                fd.write('var %s = ' % (v))
                json.dump(locals()[v], fd)
                fd.write(';')

    if args.out_data:
        print "Save into GNUplot data"
        with open(args.out_data, 'w') as fd:
            for d in ds:
                cols = map(lambda x: "%f" % (x), d['pt']) \
                        + [ str(d['l']), str(d['i']) ]
                fd.write( (" ".join(cols)) + "\n")

    if args.out_npz:
        print "Save into NPZ"
        pts = [ d['pt'] for d in ds ]
        ls = [ d['l'] for d in ds ]
        infos = [ { k:d[k] for k in ['sample', 'v', 'tr', 'src'] } for d in ds ]
        np.savez_compressed(args.out_npz, pts, ls, infos=infos)
