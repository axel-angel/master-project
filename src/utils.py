#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import rotate, shift
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
from skimage.transform import PiecewiseAffineTransform, warp
import sys
from random import randint

def flat_shape(x):
    "Returns x without singleton dimension, eg: (1,28,28) -> (28,28)"
    return x.reshape(filter(lambda s: s > 1, x.shape))

def rangesym(x1, x2, dt):
    return range(-x2, -(x1-1), dt) + range(x1, x2+1, dt)

def img_identity(i, v):
    return i
def img_shift_x(i, v):
    return shift(i, [0,v])
def img_shift_y(i, v):
    return shift(i, [v,0])
def img_blur(i, v):
    return gaussian_filter(i, v)
def img_rotate(i, v):
    return rotate(i, v, reshape=False)
def img_scale(i, v):
    return np.clip(i * v, 0, 255)

def img_sindisp_x(i, v):
    "sinusoidal displacement along x-axis"
    return rotate(img_sindisp_y(rotate(i, 90), v), -90)

def img_sindisp_y(i, v):
    "sinusoidal displacement along y-axis"
    rows, cols = i.shape[0], i.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * v
    dst_cols = src[:, 0]
    dst_rows -= v / 2
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    # because warp expect image values [0,1], we remap before/afterward
    return warp(i/255., tform, output_shape=(rows, cols))*255.


def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    xs = []
    ys = []
    for x in iterable:
        (ys if pred(x) else xs).append(x)
    return xs, ys

def parse_transfo(s):
    try:
        # format: tr:x:y or tr:x:y:z
        xs = s.split(':', 3)
        fmt = [str, int, int, int]
        return map(lambda (f,x): f(x), zip(fmt, xs))
    except:
        raise argparse.ArgumentTypeError("Invalid")

def lmdb_reader(fpath):
    lmdb_env = lmdb.open(fpath)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, flat_shape(image), label)

def npz_reader(fpath):
    npz = np.load(fpath)

    xs = npz['arr_0']
    ls = npz['arr_1']

    for i, (x, l) in enumerate(np.array([ xs, ls ]).T):
        yield (i, x, l)


def gen_adversial_random(net, img, real_label, scale):
    diffs = [ np.random.random(img.shape) * scale for _ in range(64) ]
    imgs = [ np.clip(img + d * scale * 255, 0, 255) for d in diffs ]

    imgs_caffe = np.array(imgs).reshape(64, 1, 28, 28)
    labels_caffe = np.array([ [[[0]]] ]*64) # unused

    ret = net.forward_all(data=imgs_caffe, label=labels_caffe)
    probs = ret['prob'].reshape(-1, 10)
    plabels = np.argmax(probs, axis=1)

    if np.any(plabels != real_label):
        idx = [ i for i,x in enumerate(plabels) if x != real_label ]
        for i in idx:
            yield imgs[i]


def gen_adversial(net, img, real_label, target_label,
        tries=10, scale=0.05, scale_factor = 1.5, layer = 'conv1',
        verbose=False):

    fw, bw = net.forward_backward_all(blobs=[layer], diffs=[layer],
            data=np.array([[ img ]]),
            label=np.array([[[[ target_label ]]]]))

    if verbose:
        print "Compute adversial noise (fast gradient sign method)"
    diff = np.zeros(img.shape)
    conv1_params = net.params[layer][0].data
    conv1_bw = bw[layer][0]
    for i in range(20):
     for x in range(4, 24):
      for y in range(4, 24):
       bw_ixy = conv1_bw[i,x-2,y-2]
       for u in range(5):
        for v in range(5):
         diff[x,y] -= conv1_params[i,0,4-u,4-v] * bw_ixy

    if verbose:
        print "Apply and classify"
    for _ in range(tries):
        img2 = np.clip(img + np.sign(diff) * scale * 255, 0, 255)
        ret = net.forward_all(data=np.array([[ img2 ]]),
                              label=np.array([[[[ 0 ]]]]))
        predict_label = np.argmax(ret['prob'][0])
        if verbose:
            print "Label %s not %s (scale=%f)" \
                    % (predict_label, real_label, scale)

        if predict_label != real_label:
            return { 'diff': diff, 'img': img2, 'label': real_label,
                     'plabel': predict_label, 'scale': scale }
        else:
            scale *= scale_factor

def mkCombinaisons(ranges):
    vals = map(lambda x: [x], ranges[0])
    for r in ranges[1:]:
        ys = []
        vals2 = [ xs + [y] for xs in vals for y in r ]
        vals = vals2
    return vals

def parse_transfo_grid(transfo_grid):
    trs = []
    for k, transfos in enumerate(transfo_grid):
        name = "*".join("("+ ('ALL:%s:%+i:%+i:%i' % (tr, x, y, dt)) + ")"
                        for (tr, x, y, dt) in transfos)
        ranges = [ range(x, y, dt*np.sign(y-x)) for (tr,x,y,dt) in transfos ]
        values = mkCombinaisons(ranges)
        for vs in values:
            myf = lambda i: i
            def reducer( f, (tf, v) ):
                return lambda i: tf(f(i), v)
            trfs = [ globals().get('img_%s' % (tr))
                    for (tr,x,y,dt) in transfos ]
            f = reduce(reducer, zip(trfs, vs), myf)
            trs.append({ 'f': f, 'name': name })
    return trs

def parse_transfo_random(transfo_random):
    trs = []
    def fold_transfo(f, (tr, x, y)):
        trf = globals().get('img_%s' % (tr))
        return lambda i: trf(f(i), randint(x, y))
    for transfos in transfo_random:
        name = "+".join("("+ ('RND:%s:%+i:%+i' % (tr, x, y)) + ")"
                        for (tr, x, y) in transfos)
        foldedf = reduce(fold_transfo, transfos, lambda i: i)
        trs.append({ 'f': foldedf, 'name': name })
    return trs
