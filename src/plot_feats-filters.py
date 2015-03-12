#!/usr/bin/python
import sys
import caffe
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb

def make_plot(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max() - data.min()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) \
            + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]) \
               .transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    print "Loading data"
    img = imread(args.image) # need to convert (X Y 3) -> (3 X Y)
    X = np.array([ img[:,:,0], img[:,:,1], img[:,:,2] ]) # split R/G/B

    layer = ['conv1']
    res = net.forward_all(data=np.array([ X ]), blobs=layer)

    bests = sorted(enumerate(res['prob'].reshape(-1)), key=lambda (i,p): -p)
    print "Top 5 labels:", bests[:5]

    fsshape = net.blobs[layer].data.shape[2:]
    make_plot(res['conv1'].reshape(-1, *fsshape))
