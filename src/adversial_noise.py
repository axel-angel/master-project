#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import argparse
import itertools
from utils import gen_adversial, gen_adversial_random

parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--real-label', type=int, required=True)
parser.add_argument('--target-label', type=int, required=True)
parser.add_argument('--out', type=str, default=None)
parser.add_argument('--crack', type=int, default=0)
parser.add_argument('--crack-out', type=str, default=None)
args = parser.parse_args()


n = caffe.Net(args.proto, args.model, caffe.TEST)

print "Load and forward"
img = imread(args.image, flatten=True)
res = gen_adversial(n, img, args.real_label, args.target_label, tries=10)
img2 = res['img']
scale = res['scale']

if args.out:
    if args.out == "-":
        plt.imshow(img2, interpolation='nearest', cmap='gray')
        plt.show()
    else:
        print "Save adversial figure in %s" % (args.out)
        imsave(args.out, img2)

if args.crack and args.crack_out:
    print "Found adversial (scale %f) saved in %s" % (scale, args.image)
    res = gen_adversial_random(n, img, args.real_label, scale, args.crack)

    if len(res) > 0:
        print "Save crack in %s" % (args.crack_out)
        imsave(args.crack_out, np.concatenate(res, axis=1))
    else:
        print "No crack to save, exiting"
