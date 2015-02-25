import sys
import caffe
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

MODEL_FILE = 'lenet.prototxt'
PRETRAINED = 'snapshots/lenet_mnist_v2_iter_10000.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()

print "Loading data"
X = np.load('mnist_test.npz')
datas = X['arr_0']
labels = X['arr_1']

print "Forward pass data"
dim = 500
blobs = np.array([ [0.] * dim for i in range(len(datas)) ])
for i, image in enumerate(datas):
    net.forward_all(data=np.array([[ image ]]))
    b = net.blobs['ip1'].data[0].reshape((dim,))
    blobs[i] = b.astype(np.float64)

print "Computing PCA"
pca = PCA(n_components=50)
pts_pca = pca.fit_transform(blobs)
blobs = None

print "Computing t-SNE"
tsne = TSNE(n_components=2, random_state=0)
pts = tsne.fit_transform(pts_pca)

def plot(data):
    plt.imshow(data)
    plt.show()

# take an array of shape (n, height, width) or (n, height, width, channels) and
# visualize each (height, width) thing in a grid of size approx. sqrt(n) by
# sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plot(data)
