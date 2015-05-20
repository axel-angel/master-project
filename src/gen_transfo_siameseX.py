#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import numpy as np
import argparse
import utils
from random import randint
import multiprocessing
from utils import *
from collections import defaultdict
from random import Random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--pair-displaced', action='store_true', default=False)
    args = parser.parse_args()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    ls = npz['arr_1']
    nss = npz['nss']
    assert len(ls) == len(X)
    assert len(nss) == len(X)
    count = X.shape[0]
    neighs = nss.shape[1]

    trs = [3, 6]
    print "Transformations:\n\t%s" % ("\n\t".join(map(repr, trs)))

    def process( (x, l, i) ):
        xs = []
        for v in reversed(trs):
            xs.append( (l, img_shift_x(x, -v), -v) )
        xs.append( (l, x, 0) )
        for v in trs:
            xs.append( (l, img_shift_x(x, +v), +v) )
        return (i, xs)

    print "Generate images"
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    res = {}
    Xls = np.array([ X, ls, range(count) ]).T
    itr = pool.imap_unordered(process, Xls)
    for it, (i, xs) in enumerate(itr):
        res[i] = xs
        sys.stderr.write("\rTransforming: %.0f%%" % (it * 100. / len(Xls)))
    pool.terminate()
    print ""

    def pairing_1( i ):
        imgs = []
        labels = []
        rand = Random(i)
        xs, ys, zs = rand.sample(res, k=3)
        idx = rand.randint(0, len(xs) - 2)
        # pick a similar pair
        if rand.randint(0, 1):
            # either we take the same sample with two cont transfo
            idxs = [idx, idx+1] # choose two transfo index
            (l1, i1, v1), (l2, i2, v2) = xs[idx:idx+2]
            imgs.append( np.array([ i1, i2 ]) )
            labels.append( 1 )
        else:
            # either we take two different samples but with same transfo
            # FIXME: we don't care about labels!
            idxs = [idx] # choose this single transfo index
            (l1, i1, v1), (l2, i2, v2) = xs[idx], zs[idx]
            imgs.append( np.array([ i1, i2 ]) )
            labels.append( 1 )
        # pick a disimilar pair
        (l3, i3, v3) = rand.choice([ y # pick a disimilar transfo
            for j, y in enumerate(ys) if j not in idxs ])
        imgs.append( np.array([ i1, i3 ]) )
        labels.append( 0 )

        return (i, imgs, labels)

    def pairing_2a( i ):
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        # pick similar pairs
        for idx in xrange(0, len(xs) - 1):
            # pick the sample translations
            (l1, i1, v1), (l2, i2, v2) = xs[idx:idx+2]
            imgs.append( np.array([ i1, i2 ]) )
            labels.append( 1 )
            # pick translated neighbors as well
            for n in ns:
                ys = res[n]
                for idx2 in xrange(0, len(ys)):
                    (l3, i3, v3) = ys[idx2]
                    imgs.append( np.array([ i1, i3 ]) )
                    myl = (idx == idx2) or args.pair_displaced
                    labels.append( myl )
        # pick disimilar pairs
        js = rand.sample([ k for k in xrange(count) if k not in ns ], k=neighs)
        for j in js:
            ys = res[j]
            for idx in xrange(0, len(ys) - 1):
                (l1, i1, v1) = xs[idx]
                (l4, i4, v4) = ys[idx]
                imgs.append( np.array([ i1, i4 ]) )
                labels.append( 0 )

        return (i, imgs, labels)

    def pairing_2Da( i ): # for 2D contrastive loss
        def combine_label(label_digit, label_transfo):
            return (label_digit << 0) + (label_transfo << 1)
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        # pick similar pairs
        for idx in xrange(0, len(xs) - 1):
            # pick the sample translations
            (l1, i1, v1), (l2, i2, v2) = xs[idx:idx+2]
            imgs.append( np.array([ i1, i2 ]) )
            labels.append( combine_label(1, 0) )
            # pick translated neighbors as well
            for n in ns:
                ys = res[n]
                for idx2 in xrange(0, len(ys)):
                    (l3, i3, v3) = ys[idx2]
                    imgs.append( np.array([ i1, i3 ]) )
                    labels.append( combine_label(1, int(idx == idx2)) )
        # pick disimilar pairs
        js = rand.sample([ k for k in xrange(count) if k not in ns ], k=neighs)
        for j in js:
            ys = res[j]
            for idx in xrange(0, len(ys) - 1):
                (l1, i1, v1) = xs[idx]
                (l4, i4, v4) = ys[idx]
                imgs.append( np.array([ i1, i4 ]) )
                labels.append( combine_label(int(l1 == l4), 1) )

        return (i, imgs, labels)

    def pairing_2b( i ):
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        # pick pairs
        for idx in xrange(0, len(xs) - 1):
            (l1, i1, v1) = xs[idx]
            # pick sample translations
            for idx2 in rand.sample(xrange(idx, len(xs) - 1), k=1):
                (l2, i2, v2) = xs[idx2]
                imgs.append( np.array([ i1, i2 ]) )
                labels.append( 1 )
            # pick translated neighbors as well
            for n in rand.sample(ns, k=1):
                for idx2 in rand.sample(xrange(0, len(xs)), k=1):
                    (l3, i3, v3) = res[n][idx2]
                    imgs.append( np.array([ i1, i3 ]) )
                    myl = (idx == idx2) or args.pair_displaced
                    labels.append( int(myl) )
        # pick disimilar pairs
        js = rand.sample([ k for k in xrange(count) if k not in ns ], k=len(xs))
        for j in js:
            ys = res[j]
            for idx in rand.sample(xrange(0, len(ys)), k=1):
                (l1, i1, v1) = xs[idx]
                (l4, i4, v4) = ys[idx]
                imgs.append( np.array([ i1, i4 ]) )
                labels.append( 0 )

        return (i, imgs, labels)

    out_imgs = []
    out_labels = []
    pool = multiprocessing.Pool(num_cores)
    pairingf = locals().get('pairing_'+ args.method)
    itr = pool.imap_unordered(pairingf, xrange(count))
    for it, (i, imgs, labels) in enumerate(itr):
        out_imgs.extend(imgs)
        out_labels.extend(labels)
        sys.stderr.write("\rPairing: %.0f%%" % (it * 100. / count))
    pool.terminate()

    print ""
    print "Write NPZ"
    np.savez_compressed(args.out_npz, out_imgs, out_labels)
