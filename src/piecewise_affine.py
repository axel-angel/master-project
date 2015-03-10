import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.io import imread
from sys import argv
from scipy.ndimage.interpolation import rotate
from utils import img_sindisp_x, img_sindisp_y

def transfo(i, v, axis=0):
    if axis == 0:
        return img_sindisp_x(i, v)
    else:
        return img_sindisp_y(i, v)

if __name__ == '__main__':
    imgs = map(imread, argv[1:])

    print "Compute images"
    X = []
    for j, i in enumerate(imgs):
        print "image: %i/%i" % (j, len(imgs))
        for axis in [0,1]:
            x = []
            for v in xrange(-6, 7, 2):
                print "\ttransfo: axis=%i v=%i" % (axis, v)
                x.append(transfo(i, v, axis=axis))
            X.append(x)

    print "Prepare plot"
    for i, x in enumerate(X):
        X[i] = np.concatenate(x, axis=1)
    Y = np.concatenate(X, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(Y, cmap='gray', interpolation='nearest')
    #ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    #ax.axis((0, cols, rows, 0))
    plt.show()
