#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import scipy.ndimage
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--fwd-npz', type=str, required=True)
    parser.add_argument('--pca-npz', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    args = parser.parse_args()

    print "Load data"
    imgs_orig = [ scipy.ndimage.imread(i, flatten=True).astype(np.uint8)
                for i in args.image ]
    fwd_npz = np.load(args.fwd_npz)
    blobs = fwd_npz[args.layer]
    labels = fwd_npz['labels']
    infos = np.array([ dict(src="dataset", l=l, tr="identity", v=0)
                     for l in labels ])

    pca_npz = np.load(args.pca_npz)
    pca = pca_npz['pca'].flat.next()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Compute transformations"
    imgs_tr = []
    for i, img in enumerate(imgs_orig):
        print "  input image:", args.image[i]
        for k, t in transformations.iteritems():
            print "    transform:", k
            for s in t['steps']:
                imgs_tr.append(t['f'](img, s))
                fname = "".join(args.image[i].split('/')[-1].split('.')[:-1])
                infos = np.append(infos, dict(src=fname, l=-1, tr=k, v=s))

    print "Transform forward output"
    imgs_tr_np = np.array(imgs_tr).reshape(-1, 1, 28, 28)
    res = net.forward_all(data=imgs_tr_np, blobs=[args.layer])
    iblobs = res[args.layer].reshape(len(imgs_tr), -1)
    blobs = np.concatenate((blobs, iblobs))
    blobs_pca = pca.transform(blobs)

    print "Computing t-SNE"
    tsne = TSNE(n_components=2, random_state=0, verbose=True)
    pts2 = tsne.fit_transform(blobs_pca)

    print "Dump into npz"
    np.savez(args.out_npz, imgs_tr_np=imgs_tr_np, tsne=tsne, pts=pts2,
            infos=infos)
