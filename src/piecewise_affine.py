import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.io import imread
from sys import argv
from scipy.ndimage.interpolation import rotate

axis = 1

def transfo(i, v, axis=0):
    if axis == 0:
        i = rotate(i, 90)

    rows, cols = i.shape[0], i.shape[1]

    src_cols = np.linspace(0, cols, 28)
    src_rows = np.linspace(0, rows, 28)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * v
    dst_cols = src[:, 0]
    dst_rows -= v / 2
    dst = np.vstack([dst_cols, dst_rows]).T


    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out = warp(i, tform, output_shape=(rows, cols))
    if axis == 0:
        out = rotate(out, -90)
    return out

if __name__ == '__main__':
    imgs = map(imread, argv[1:])

    X = []
    for i in imgs:
        for axis in [0,1]:
            x = []
            for v in xrange(-6, 7, 2):
                x.append(transfo(i, v, axis=axis))
            X.append(x)

    for i, x in enumerate(X):
        X[i] = np.concatenate(x, axis=1)
    Y = np.concatenate(X, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(Y, cmap='gray', interpolation='nearest')
    #ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    #ax.axis((0, cols, rows, 0))
    plt.show()
