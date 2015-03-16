#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from random import random
from math import sin, cos

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--filter', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    print "Load data"
    npz = np.load(args.in_npz)

    imgs_tr_np = npz['imgs_tr_np']
    tsne = npz['tsne']
    pts = npz['pts']
    infos = npz['infos']

    label_set = set(i['l'] for i in infos if i['src'] == "dataset")
    trans_set = set(i['tr'] for i in infos if i['src'] != "dataset")
    label_max = max(label_set)
    tr_map = { k:label_max+1+i for i, k in enumerate(trans_set) }

    print "Filtering data"
    Y = np.array([pts, infos]).T
    Y2 = filter(lambda y: args.filter in y[1]['src'], Y)

    ds_annotates = set()
    for [pt, info] in Y:
        label = info['l']
        if info['src'] == "dataset" and label not in ds_annotates:
            Y2.append([ pt, info ])
            ds_annotates.add(label)

    print "Make plot"
    colors = np.array([ i['l'] if i['l'] >= 0 else tr_map[i['tr']]
                      for [pt, i] in Y ]).astype(np.uint8)
    plt.figure(figsize=(30, 30))
    plt.scatter(pts[:,0], pts[:,1], s=5, cmap='bwr', c=colors,
            edgecolors='none')
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5)
    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', linewidth=0.2)
    for i, [pt, info] in enumerate(Y2):
        label = info['l']
        tsize = 40 + random() * 5
        tx = tsize * cos(i / 4.)
        ty = tsize * sin(i / 4.)

        if info['v'] == 0:
            txt = "D%s" % (label)
        else:
            txt = "I%s_%s(%i)" % (info['src'], info['tr'], info['v'])

        plt.annotate(
            txt,
            xy=pt, xytext=(tx, ty),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = None, arrowprops = arrowprops,
            size = '5',
        )

    plt.savefig(args.out, bbox_inches='tight', dpi=300)
