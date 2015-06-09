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
from itertools import izip, imap
from operator import neg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--pair-displaced', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--shuffle-seed', type=int, default=42)
    parser.add_argument('--grouped', type=int, default=None)
    parser.add_argument('--transfo-values', type=int, nargs='+')
    parser.add_argument('--transfo-identity', type=int, default=0)
    parser.add_argument('--transfo-name', type=str, required=True)
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

    tr_f = globals().get('img_%s' % (args.transfo_name))
    trs_all = args.transfo_values
    #trs = [3, 6]
    idx_origs = filter(lambda (i,v): v == args.transfo_identity,
            enumerate(trs_all))
    assert len(idx_origs) == 1
    idx_orig = idx_origs[0][0] # index of identity transfo value in trs_all
    print "Transformations: %s, [%s]" % (tr_f, ", ".join(map(repr, trs_all)))

    def process( (x, l, i) ):
        xs = []
        for v in trs_all:
            xs.append( (l, tr_f(x, v), v) )
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

    def combine_label(label_digit, label_transfo):
        return (label_digit << 0) + (label_transfo << 1)

    def pairing_1( i ):
        imgs = []
        labels = []
        rand = Random(i)
        xs, ys, zs = map(res.get, rand.sample(res, k=3))
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
        for idx in xrange(0, len(xs) - 2):
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
            for idx in xrange(0, len(ys)):
                (l1, i1, v1) = xs[idx]
                (l4, i4, v4) = ys[idx]
                imgs.append( np.array([ i1, i4 ]) )
                labels.append( combine_label(int(l1 == l4), 1) )

        return (i, imgs, labels)

    def pairing_2Db( i ): # for 2D contrastive loss
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        (l, _, _) = xs[0] # label
        yss = rand.sample(filter(lambda ys: ys[0][0] != l, res.itervalues()),
                k=neighs) # dissimilar labels
        for idx in xrange(0, len(xs)):
            (l1, i1, v1) = xs[idx]
            # pick similar pairs
            for n in ns:
                ys = res[n]
                idx2 = rand.randint(0, len(ys) - 1)
                (l3, i3, v3) = ys[idx2]
                imgs.append( np.array([ i1, i3 ]) )
                labels.append( combine_label(1, int(idx == idx2)) )
            # pick disimilar pairs
            for ys in yss:
                (l5, i5, v5) = ys[idx]
                imgs.append( np.array([ i1, i5 ]) )
                labels.append( combine_label(0, 1) )

                idx2s = set(xrange(0, len(ys)))
                idx2s.remove(idx)
                idx2 = rand.choice(list(idx2s))
                (l4, i4, v4) = ys[idx2]
                imgs.append( np.array([ i1, i4 ]) )
                labels.append( combine_label(0, 0) )

        return (i, imgs, labels)

    def pairing_2Dc( i ): # for 2D contrastive loss
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        (l, _, _) = xs[0] # label
        yss = rand.sample(filter(lambda ys: ys[0][0] != l, res.itervalues()),
                k=neighs) # dissimilar labels
        for idx in xrange(0, len(xs)):
            (l1, i1, v1) = xs[idx]
            # pick similar pairs
            idx2s = filter(lambda idx2: idx2 != idx, xrange(0, len(xs)))
            for idx2 in idx2s:
                (l2, i2, v2) = xs[idx2]
                assert v1 != v2
                imgs.append( np.array([ i1, i2 ]) )
                labels.append( combine_label(1, 0) )
            for n in ns:
                ys = res[n]
                (l3, i3, v3) = ys[idx]
                assert v1 == v3
                imgs.append( np.array([ i1, i3 ]) )
                labels.append( combine_label(1, 1) )
            # pick disimilar pairs
            for ys in yss:
                (l5, i5, v5) = ys[idx]
                imgs.append( np.array([ i1, i5 ]) )
                labels.append( combine_label(0, 1) )

                idx2s = set(xrange(0, len(ys)))
                idx2s.remove(idx)
                idx2 = rand.choice(list(idx2s))
                (l4, i4, v4) = ys[idx2]
                imgs.append( np.array([ i1, i4 ]) )
                labels.append( combine_label(0, 0) )

        return (i, imgs, labels)

    def pairing_2Dd( i ): # for 2D contrastive loss
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        (l, _, _) = xs[0] # label
        yss = rand.sample(filter(lambda ys: ys[0][0] != l, res.itervalues()),
                k=neighs) # dissimilar labels
        for idx in xrange(0, len(xs) - 1):
            (l1, i1, v1) = xs[idx]
            # pick similar pairs
            idx2s = filter(lambda idx2: idx2 != idx, xrange(0, len(xs) - 1))
            for idx2 in idx2s:
                (l2, i2, v2) = xs[idx2]
                assert v1 != v2
                imgs.append( np.array([ i1, i2 ]) )
                labels.append( combine_label(1, 0) )
            # no neighbor pairing here
            # pick disimilar pairs
            for ys in yss:
                (l5, i5, v5) = ys[idx]
                imgs.append( np.array([ i1, i5 ]) )
                labels.append( combine_label(0, 1) )

                idx2s = set(xrange(0, len(ys) - 1))
                idx2s.remove(idx)
                idx2 = rand.choice(list(idx2s))
                (l4, i4, v4) = ys[idx2]
                imgs.append( np.array([ i1, i4 ]) )
                labels.append( combine_label(0, 0) )

        return (i, imgs, labels)

    def pairing_2b( i ):
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        (l0, i0, v0) = xs[idx_orig]
        # pick pairs
        for idx in xrange(0, len(xs)):
            (l1, i1, v1) = xs[idx]
            # pair original with its 4 translations
            if idx != idx_orig:
                imgs.append( np.array([ i0, i1 ]) )
                labels.append( 1 )
            # pair all 4 translations of 5 neighbors
            for n in ns:
                for idx2 in xrange(0, len(xs)):
                    (l2, i2, v2) = res[n][idx2]
                    imgs.append( np.array([ i1, i2 ]) )
                    myl = (idx == idx2) or args.pair_displaced
                    labels.append( int(myl) )
        # pick lots of disimilar pairs
        js = rand.sample([ k for k in xrange(count) if k not in ns ], k=100)
        for j in js:
            ys = res[j]
            idx = rand.randint(0, len(xs) - 1)
            idx2 = rand.randint(0, len(ys) - 1)
            (l1, i1, v1) = xs[idx]
            (l3, i3, v3) = ys[idx2]
            imgs.append( np.array([ i1, i3 ]) )
            labels.append( 0 )

        return (i, imgs, labels)

    def pairing_0( i ):
        imgs = []
        labels = []
        rand = Random(i)
        xs = res[i]
        ns = nss[i]
        (l1, i1, v1) = xs[idx_orig]
        # pick pairs, only neighbors
        for n in ns:
            (l2, i2, v2) = res[n][idx_orig]
            imgs.append( np.array([ i1, i2 ]) )
            labels.append( 1 )
        # pick disimilar pairs
        js = rand.sample([ k for k in xrange(count) if k not in ns ], k=neighs)
        for j in js:
            ys = res[j]
            (l1, i1, v1) = xs[idx_orig]
            (l3, i3, v3) = ys[idx_orig]
            imgs.append( np.array([ i1, i3 ]) )
            labels.append( 0 )

        return (i, imgs, labels)

    out_imgs = []
    out_labels = []
    pool = multiprocessing.Pool(num_cores)
    pairingf = locals().get('pairing_'+ args.method)
    itr = pool.imap_unordered(pairingf, xrange(count))
    #itr = imap(pairingf, xrange(count))
    for it, (i, imgs, labels) in enumerate(itr):
        out_imgs.extend(imgs)
        out_labels.extend(labels)
        sys.stderr.write("\rPairing: %.0f%%" % (it * 100. / count))
    pool.terminate()
    print ""

    assert len(out_imgs) == len(out_labels)
    if args.shuffle:
        print "Shuffle dataset"
        Random(args.shuffle_seed).shuffle(out_imgs)
        Random(args.shuffle_seed).shuffle(out_labels)

    if args.grouped:
        print "Group per label by %i" % (args.grouped)
        out_d = defaultdict(list)
        for x, l in izip(out_imgs, out_labels):
            out_d[l].append(x)
        print "  has: %s" % ({ l:len(xs) for l,xs in out_d.iteritems() })
        out_imgs = []
        out_labels = []
        # for each label, make lazy chunks of grouped-length
        out_iters = { l: izip(*([ iter(out_d[l]) ] * args.grouped))
                for l in out_d }
        counts = defaultdict(int)
        try:
            while True:
                for l in out_d.iterkeys():
                    xs = out_iters[l].next()
                    out_imgs.extend(xs)
                    out_labels.extend([l] * len(xs))
                    counts[l] += len(xs)
                    assert len(out_imgs) == len(out_labels)
        except StopIteration:
            pass
        print "  left: %s" % ({ l:len(out_d[l])-counts[l] for l in out_d })

    assert len(out_imgs) == len(out_labels)
    print "Write NPZ, %i pairs" % (len(out_imgs))
    np.savez_compressed(args.out_npz, out_imgs, out_labels)
