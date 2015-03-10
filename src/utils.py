#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import rotate, shift
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
from skimage.transform import PiecewiseAffineTransform, warp

def img_shift_x(i, v):
    return shift(i, (v,0))
def img_shift_y(i, v):
    return shift(i, (0,v))
def img_blur(i, v):
    return gaussian_filter(i, v)
def img_rotate(i, v):
    return rotate(i, v, reshape=False)
def img_scale(i, v):
    return np.clip(i * v, 0, 255)

def img_sindisp_x(i, v):
    "sinusoidal displacement along x-axis"
    return rotate(img_sindisp_y(rotate(i, 90), v), -90)

def img_sindisp_y(i, v):
    "sinusoidal displacement along y-axis"
    rows, cols = i.shape[0], i.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * v
    dst_cols = src[:, 0]
    dst_rows -= v / 2
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    return warp(i, tform, output_shape=(rows, cols))
