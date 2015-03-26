import sys
import caffe
import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'bwr'

MODEL_FILE = 'lenet.prototxt'
PRETRAINED = 'snapshots/lenet_mnist_v2_iter_10000.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()

print "Loading data"
X = np.load('mnist_test.npz')
datas = X['arr_0']
labels = X['arr_1']

print "Loaded images, shape %s" % (str(datas.shape))

print "Forward pass data"
dim = 500
layer = 'ip1'
# TODO: rewrite this with a single forward (with layer=[…])
blobs = np.array([ [0.] * dim for i in range(len(datas)) ])
for i, image in enumerate(datas):
    net.forward_all(data=np.array([[ image ]]))
    b = net.blobs[layer].data[0].reshape((dim,))
    blobs[i] = b.astype(np.float64)

# TODO: save output in a file for later reusability

print "Computing PCA"
pca = PCA(n_components=50)
pts_pca = pca.fit_transform(blobs)
blobs = None

print "Computing t-SNE"
tsne = TSNE(n_components=2, random_state=0)
pts = tsne.fit_transform(pts_pca)

now = time.time()
fout = 't-sne_%ik_%s_%i' % (pts.shape[0]/1000, layer, now)
fout_npz = 'save_%s.npz' % (fout)
fout_png = 'plots/t-sne/%s.png' % (fout)

np.savez_compressed(fout_npz, pca=pca, tsne=tsne, pts=pts, labels=labels)
print "Saved t-SNE in %s" % (fout_npz)

colors = labels
plt.scatter(pts[:,0], pts[:,1], s=25, c=colors)
plt.axis('off')
plt.savefig(fout_png, bbox_inches='tight')
print "Ploted image in %s" % (fout_png)
