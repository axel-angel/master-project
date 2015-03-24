#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from random import random
from math import sin, cos
from utils import partition
import json
from collections import defaultdict
import lmdb
import caffe
from base64 import b64encode
from scipy.misc import imsave
from io import BytesIO
from utils import *

def conv(d): # legacy conversion of label to int
    d['l'] = int(d['l'])
    return d

def img_encode(i):
    bio = BytesIO()
    imsave(bio, flat_shape(img.astype(np.uint8)), format='jpeg')
    return b64encode(bio.getvalue())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-npz', type=str, required=True)
    parser.add_argument('--out-js', type=str, required=True)
    parser.add_argument('--lmdb', type=str, required=True)
    args = parser.parse_args()

    X = np.load(args.in_npz)
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
    lmdb_env = lmdb.open(args.lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    imgs = []
    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        img = caffe.io.datum_to_array(datum)
        imgs.append(img_encode(img)) # include dataset set
    for img in X['imgs_tr_np']:
        imgs.append(img_encode(img)) # include distorted set too

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
