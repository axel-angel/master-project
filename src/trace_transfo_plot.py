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
    parser.add_argument('--transfo', type=str, default=None)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    print "Load data"
    npz = np.load(args.in_npz)

    imgs_tr_np = npz['imgs_tr_np']
    tsne = npz['tsne']
    pts = npz['pts']
    labels = npz['labels']
    infos = npz['infos']
    tr_map = npz['tr_map'].flat.next()

    print "Filtering data"
    Y = np.array([pts, labels, infos]).T
    if args.transfo:
        Y2 = np.array(filter(lambda x: x[1] == tr_map[args.transfo], Y))
    else:
        Y2 = np.array(filter(lambda x: x[2] != "dataset", Y))

    print "Make plot"
    plt.figure(figsize=(30, 30))
    plt.scatter(pts[:,0], pts[:,1], s=5, cmap='bwr', c=labels, edgecolors='none')
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5)
    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0', linewidth=0.2)
    for i, [pt, label, info] in enumerate(Y2):
        tsize = 40 + random() * 5
        tx = tsize * cos(i / 4.)
        ty = tsize * sin(i / 4.)
        plt.annotate(
            info,
            xy=pt, xytext=(tx, ty),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = None, arrowprops = arrowprops,
            size = '5',
        )

    plt.savefig(args.out, bbox_inches='tight', dpi=300)
