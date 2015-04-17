#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--real-label', type=int, required=True)
parser.add_argument('--target-label', type=int, required=True)
parser.add_argument('--out', type=str, default=None)
args = parser.parse_args()

n = caffe.Net(args.proto, args.model, caffe.TEST)

real_label = args.real_label
adversial_label = args.target_label

print "Load and forward"
img = imread(args.image, flatten=True)
fw, bw = n.forward_backward_all(blobs=['conv1'], diffs=['conv1'],
        data=np.array([[ img ]]), label=np.array([[[[ adversial_label ]]]]))

# FIXME: Need to optimize this computations! (use vector/matrix)
print "Compute adversial noise (fast gradient sign method)"
diff = np.zeros(img.shape)
conv1_params = n.params['conv1'][0].data
conv1_bw = bw['conv1'][0]
for i in range(20):
 for x in range(4, 24):
  for y in range(4, 24):
   bw_ixy = conv1_bw[i,x-2,y-2]
   for u in range(5):
    for v in range(5):
     diff[x,y] -= conv1_params[i,0,4-u,4-v] * bw_ixy

print "Apply and classify"
scale = 0.05
for _ in range(10):
    img2 = np.clip(img + np.sign(diff) * scale * 255, 0, 255)
    ret = n.forward_all(data=np.array([[ img2 ]]), label=np.array([[[[ 0 ]]]]))
    pl = np.argmax(ret['prob'][0])
    print "Label", pl, "for scale", scale

    worked = pl != real_label
    if worked:
        break
    else:
        scale *= 1.5

if args.out:
    if args.out == "-":
        plt.imshow(img2, interpolation='nearest', cmap='gray')
        plt.show()
    else:
        print "Save adversial figure in %s" % (args.out)
        imsave(args.out, img2)
