#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--real-label', type=int, required=True)
parser.add_argument('--target-label', type=int, required=True)
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
for x in range(4, 24):
 for y in range(4, 24):
  for i in range(20):
   for u in range(5):
    for v in range(5):
     diff[x,y] -= n.params['conv1'][0].data[i,0,4-u,4-v] \
                * bw['conv1'][0,i,x-2,y-2]

print "Apply and classify"
worked = False
scale = 0.05
while not worked:
    img2 = np.clip(img + np.sign(diff) * scale * 255, 0, 255)
    ret = n.forward_all(data=np.array([[ img2 ]]), label=np.array([[[[ 0 ]]]]))
    pl = np.argmax(ret['prob'][0])
    print "Scale", scale
    print "Label:", pl
    print "Loss:", ret['loss']
    worked = pl != real_label
    scale *= 2
    if scale >= 10:
        print "Failing, abort!"
        exit()

plt.imshow(img2, interpolation='nearest', cmap='gray');
plt.show()
