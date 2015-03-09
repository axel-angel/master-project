#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import scipy.ndimage
import utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

layer = 'ip1'
transformations = {
    "shift_x": { "f": utils.img_shift_x, "steps": range(-15, 16, 3), },
    "shift_y": { "f": utils.img_shift_y, "steps": range(-15, 16, 3), },
    "blur": { "f": utils.img_blur, "steps": range(0, 4, 1) },
    "rotate": { "f": utils.img_rotate, "steps": range(-45, 46, 10) },
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--ip1-npz', type=str, required=True)
    parser.add_argument('--tsne-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    args = parser.parse_args()

    print "Load data"
    img_orig = scipy.ndimage.imread(args.image, flatten=True)
    ip1 = np.load(args.ip1_npz)['blobs']

    X = np.load(args.tsne_npz)
    pca = X['pca'].flat.next()
    pts = X['pts'] # assume already PCA-transformed
    labels = X['labels'] * 0 # FIXME: same label for old points

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    tr_map = { k:1+i for i, k in enumerate(transformations.keys()) }
    imgs_tr = [ img_orig ]
    labels = np.append(labels,  [ 0 ])

    print "Compute transformations"
    for k, t in transformations.iteritems():
        print "\ttransform:", k
        for s in t['steps']:
            imgs_tr.append(t['f'](img_orig, s))
            labels = np.append(labels, tr_map[k])

    print "Compute forward output"
    imgs_tr_np = np.array(imgs_tr).reshape(-1, 1, 28, 28)
    res = net.forward_all(data=imgs_tr_np, blobs=[layer])
    iblob = res[layer].reshape(len(imgs_tr), -1)

    print "Transform forward output"
    iblob_pca = pca.transform(iblob)

    print "Computing t-SNE"
    X = np.append(pts, iblob_pca)
    tsne = TSNE(n_components=2, random_state=0, verbose=True)
    pts2 = tsne.fit_transform(X)

    print "Save and plot"
    np.savez(args.out_npz, imgs_tr_np=imgs_tr_np, tsne=tsne, pts=pts2)
