#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import scipy.ndimage
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

layer = 'ip1'
transformations = {
    "identity": { "f": lambda i,v: i, "steps": [0], },
    "shift_x": { "f": img_shift_x, "steps": rangesym(1, 15, 2) },
    "shift_y": { "f": img_shift_y, "steps": rangesym(1, 25, 2) },
    "blur": { "f": img_blur, "steps": range(1, 5, 1) },
    "rotate": { "f": img_rotate, "steps": rangesym(1, 93, 2) },
    "sindisp_x": { "f": img_sindisp_x, "steps": rangesym(1, 6, 1) },
    "sindisp_y": { "f": img_sindisp_y, "steps": rangesym(1, 6, 1) },
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, nargs='+')
    parser.add_argument('--ip1-npz', type=str, required=True)
    parser.add_argument('--tsne-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    args = parser.parse_args()

    print "Load data"
    imgs_orig = [ scipy.ndimage.imread(i, flatten=True).astype(np.uint8)
                for i in args.image ]
    ip1_npz = np.load(args.ip1_npz)
    blobs = ip1_npz['blobs']
    labels = ip1_npz['labels']
    label_max = labels.max()
    infos = np.array([ dict(input="dataset %i" % (l), tr="identity", v=0)
                     for l in labels ])

    tsne_npz = np.load(args.tsne_npz)
    pca = tsne_npz['pca'].flat.next()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    tr_map = { k:label_max+1+i for i, k in enumerate(transformations.keys()) }

    print "Compute transformations"
    imgs_tr = []
    for i, img in enumerate(imgs_orig):
        print "  input image:", args.image[i]
        for k, t in transformations.iteritems():
            print "    transform:", k, tr_map[k]
            for s in t['steps']:
                imgs_tr.append(t['f'](img, s))
                labels = np.append(labels, tr_map[k])
                infos = np.append(infos, dict(input=args.image[i], tr=k, v=s))

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
