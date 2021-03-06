#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import matplotlib.pyplot as plt
from utils import partition

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsne-npz', type=str, required=True)
    parser.add_argument('--data-npz', type=str, required=True)
    parser.add_argument('--filter', type=str, default='')
    parser.add_argument('--top', type=int, default=1)
    parser.add_argument('--dump-top', type=str, default=None)
    parser.add_argument('--transfo', type=str, nargs='*', default=[])
    parser.add_argument('--value', type=int, nargs='*', default=[])
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    # X: [array([-0.36007209, -3.17370899]), {'src': '9', 'tr': 'identity', 'l': -1, 'v': 0}, 394]], dtype=object)

    print "Load data"
    tsne = np.load(args.tsne_npz)
    X = np.array([ tsne['pts'], tsne['infos'], range(len(tsne['pts'])) ]).T
    imgs_tr = tsne['imgs_tr_np']
    n_dim = imgs_tr.shape[-2:]

    data_npz = np.load(args.data_npz)
    data_imgs = data_npz['arr_0']

    Xn, Xd = partition(lambda y: y[1]['src'] == "dataset", X)
    Y2 = filter(lambda y: args.filter in y[1]['src'], Xn)
    dataset_len = len(Xd)

    entries = []
    for pt, i, j in Y2:
        if len(args.transfo) > 0 and i['tr'] not in args.transfo:
            continue
        if len(args.value) > 0 and i['v'] not in args.value:
            continue

        d = {}
        d['orig'] = [pt, i, j]

        print "Closest to %s (%i/%i): %s" % (pt, j, j - dataset_len, i)
        xs = sorted(Xd, key=lambda y: np.sum((y[0] - pt) ** 2))[:args.top]
        for pt, i, j in xs:
            print "  %s (%i): %s" % (pt, j, i)
        d['ns'] = xs

        entries.append(d)

    imgs = np.zeros((len(entries), 1+args.top) + n_dim)
    for i0, e in enumerate(entries):
        imgs[i0, 0] = imgs_tr[e['orig'][2] - dataset_len]
        for i1, (_, _, j) in enumerate(e['ns']):
            imgs[i0, 1+i1] = data_imgs[j]

    if args.dump_top:
        print "Dump closest to %s" % (args.dump_top)
        np.savez_compressed(args.dump_top, entries=entries, imgs=imgs)

    Y = np.concatenate([ np.concatenate(i, axis=0) for i in imgs ], axis=1)
    plt.imshow(Y, cmap='gray', interpolation='nearest')
    if args.out == "-":
        plt.show()
    else:
        plt.savefig(args.out, bbox_inches='tight', dpi=300)
