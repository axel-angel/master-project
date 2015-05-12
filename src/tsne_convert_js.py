#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import json
from collections import defaultdict
from utils import *

def conv(d): # legacy conversion of label to int
    d['l'] = int(d['l'])
    return d

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsne-npz', type=str, required=True)
    parser.add_argument('--out-js', type=str, required=True)
    parser.add_argument('--dataset-npz', type=str, required=True)
    args = parser.parse_args()

    X = np.load(args.tsne_npz)
    Y = np.array([ X['pts'], X['infos'], range(X['pts'].shape[0]) ]).T

    # stats per src, label, transformations
    src_set = defaultdict(int)
    label_set = defaultdict(int)
    tr_set = defaultdict(int)
    for _, info, _ in Y:
        src_set[info['src']] += 1
        label_set[str(info['l'])] += 1
        tr_set[info['tr']] += 1

    # load dataset images (encode in base64 for easy HTML inclusion)
    dataset_npz = np.load(args.dataset_npz)
    imgs = []
    for img in dataset_npz['arr_0']:
        imgs.append(js_img_encode(img)) # include dataset set
    for img in X['imgs_tr_np']:
        imgs.append(js_img_encode(img)) # include distorted set too

    # convert to json-serialisable
    ds = [ dict(x=pt[0], y=pt[1], i=i, **conv(info)) for pt, info, i in Y ]
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
