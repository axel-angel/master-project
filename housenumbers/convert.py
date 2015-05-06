#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import imresize
from scipy.io import loadmat
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--in-mat', type=str, required=True)
parser.add_argument('--out-npz', type=str, required=True)
args = parser.parse_args()

# from wikipedia: Converting color to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def digitWhite(g):
    "try to make digit white on black, guess with average color"
    if np.average(g) >= 127:
        return 255 - g
    else:
        return g

print "Load data"
npz = loadmat(args.in_mat)
y = npz['y'].reshape(-1)
X = np.transpose(npz['X'], axes=[3,0,1,2])

print "Convert, resize"
def process(x):
    return imresize(digitWhite(rgb2gray(x)), size=(28,28))

num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)
y2 = np.array([ l % 10 for l in y ])
X2 = np.array(list(pool.map(process, X)))
pool.terminate()

print "Save"
np.savez_compressed(args.out_npz, X2, y2)
