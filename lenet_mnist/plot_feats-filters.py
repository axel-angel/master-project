#!/usr/bin/python
import sys
import caffe
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

MODEL_FILE = 'lenet.prototxt'
PRETRAINED = 'snapshots/lenet_mnist_v2_iter_10000.caffemodel'
PLOT_FILTERS_DIR = "plots/filters"
PLOT_FEATURES_DIR = "plots/features"

# inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb

def make_plot(data, filename, padsize=1, padval=0):
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
    plt.savefig(filename, bbox_inches='tight')


# -- main
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()

print "Loading data"
X = np.load('mnist_test.npz')
datas = X['arr_0']
labels = X['arr_1']

print "Loaded images, shape %s" % (str(datas.shape))

samples = np.zeros(shape=(10, 1, 28, 28), dtype=np.uint8)
for i in xrange(0, 10):
    j = next(u for u, x in np.ndenumerate(labels) if x == i)
    samples[i][0] = datas[j]

print "Forward pass data"
blobs = ['conv1', 'conv2']
bs = net.forward_all(data=samples, blobs=blobs)

print "Plotting filters"
for b in blobs:
    # filters grouped by input feature map
    f = net.params[b][0].data.transpose((1, 0, 2, 3))
    for i in xrange(0, f.shape[0]):
        print "filters %s: layer %i" % (b, i)
        make_plot(f[i], "%s/%s_%02i.png" % (PLOT_FILTERS_DIR, b, i))


print "Plotting features"
for i in xrange(0, samples.shape[0]):
    for b in blobs:
        print "features %i: layer %s" % (i, b)
        make_plot(bs[b][i], "%s/%s_%i.png" % (PLOT_FEATURES_DIR, b, i))
