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
    "shift_x": { "f": utils.img_shift_x, "steps": range(-15, 16, 2), },
    "shift_y": { "f": utils.img_shift_y, "steps": range(-25, 26, 2), },
    "blur": { "f": utils.img_blur, "steps": range(0, 4, 1) },
    "rotate": { "f": utils.img_rotate, "steps": range(-60, 61, 2) },
    "sindisp_x": { "f": utils.img_sindisp_x, "steps": range(-6, 7, 1) },
    "sindisp_y": { "f": utils.img_sindisp_y, "steps": range(-6, 7, 1) },
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
    img_orig = scipy.ndimage.imread(args.image, flatten=True).astype(np.uint8)
    ip1_npz = np.load(args.ip1_npz)
    blobs = ip1_npz['blobs']
    labels = ip1_npz['labels']
    label_max = labels.max()
    infos = np.repeat([ "dataset" ], labels.shape[0])

    tsne_npz = np.load(args.tsne_npz)
    pca = tsne_npz['pca'].flat.next()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    tr_map = { k:label_max+1+i for i, k in enumerate(transformations.keys()) }
    imgs_tr = [ img_orig ]
    labels = np.append(labels,  [ 0 ])
    infos = np.append(infos, "input")

    print "Compute transformations"
    for k, t in transformations.iteritems():
        print "\ttransform:", k, tr_map[k]
        for s in t['steps']:
            imgs_tr.append(t['f'](img_orig, s))
            labels = np.append(labels, tr_map[k])
            infos = np.append(infos, "input %s %i" % (k, s))

    print "Transform forward output"
    imgs_tr_np = np.array(imgs_tr).reshape(-1, 1, 28, 28)
    res = net.forward_all(data=imgs_tr_np, blobs=[layer])
    iblobs = res[layer].reshape(len(imgs_tr), -1)
    blobs = np.concatenate((blobs, iblobs))
    blobs_pca = pca.transform(blobs)

    print "Computing t-SNE"
    tsne = TSNE(n_components=2, random_state=0, verbose=True)
    pts2 = tsne.fit_transform(blobs_pca)

    print "Dump into npz"
    np.savez(args.out_npz, imgs_tr_np=imgs_tr_np, tsne=tsne, pts=pts2,
            labels=labels, infos=infos, tr_map=tr_map)
