#!/usr/bin/python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

n = caffe.Net('lenet_loss.prototxt', 'snapshots/lenet_mnist_v5-orig_iter_10000.caffemodel', caffe.TEST)

print "Load and forward"
img = imread('data/test1/0a.png', flatten=True)
fw, bw = n.forward_backward_all(blobs=['conv1'], diffs=['conv1'], data=np.array([[ img ]]), label=np.array([[[[ 8 ]]]]))

print "Compute adversial noise (fast gradient sign method)"
diff = np.zeros(img.shape)
for x in range(4, 24):
 for y in range(4, 24):
  for i in range(20):
   for u in range(5):
    for v in range(5):
     diff[x,y] += n.params['conv1'][0].data[i,0,4-u,4-v] * bw['conv1'][0,i,x-2,y-2]

print "Apply and classify"
worked = False
scale = 0.25
while not worked:
    img2 = np.clip(img + np.sign(diff) * scale * 255, 0, 255)
    ret = n.forward_all(data=np.array([[ img2 ]]), label=np.array([[[[ 1 ]]]]))
    pl = np.argmax(ret['prob'][0])
    print "Scale", scale
    print "Label:", pl
    print "Loss:", ret['loss']
    worked = pl != 0
    scale *= 1.5
