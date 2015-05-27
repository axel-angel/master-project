#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import numpy as np
import argparse
from random import randint
from collections import defaultdict
from random import Random
from itertools import izip, imap, product

def combine_label(label_digit, label_transfo):
    return (label_digit << 0) + (label_transfo << 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--grouped', type=int, default=None)
    args = parser.parse_args()

    print "Load dataset"
    npz = np.load(args.in_npz)
    X = npz['arr_0']
    infos = npz['arr_2']
    assert len(infos) == len(X)
    count = X.shape[0]

    # just focus on elevation and azimuth
    xss = defaultdict(list)
    for x, ii in izip(X, infos):
        elv = ii['elevation']
        azi = ii['azimuth']
        xss[(elv,azi)].append(x)

    # hard-coded NORB ranges
    elvs = range(0, 9) # 0 to 8
    azis = range(0, 35, 2) # 0 to 34

    print "Elevation:", elvs
    print "Azimuth:", azis

    out_imgs = []
    out_labels = []
    rand = Random(42)
    eas = product(elvs, azis, elvs, azis)
    eas_count = len(elvs)**2 * len(azis)**2
    for it, (elv1, azi1, elv2, azi2) in enumerate(eas):
        if (elv1 >= elv2) or (azi1 >= azi2):
            continue
        xs1 = xss[(elv1,azi1)]
        xs2 = xss[(elv2,azi2)]
        isneigh_elv = (abs(elv1 - elv2) == 1)
        isneigh_azi = (abs(azi1 - azi2) == 2) or (abs(azi1 - azi2) == 34)
        isneigh = isneigh_elv and isneigh_azi
        for x1, x2 in product(xs1, xs2):
            out_imgs.append( np.array([ x1, x2 ]) )
            out_labels.append( int(isneigh) )
        sys.stderr.write("\rPairing: %.0f%%" % (it * 100. / eas_count))
    print ""

    assert len(out_imgs) == len(out_labels)
    if args.shuffle:
        print "Shuffle dataset"
        Random(42).shuffle(out_imgs)
        Random(42).shuffle(out_labels)

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
