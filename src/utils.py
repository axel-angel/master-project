#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage
import scipy.misc

def img_shift_x(i, v):
    return scipy.ndimage.interpolation.shift(i, (v,0))
def img_shift_y(i, v):
    return scipy.ndimage.interpolation.shift(i, (0,v))
def img_blur(i, v):
    return scipy.ndimage.filters.gaussian_filter(i, v)
def img_rotate(i, v):
    return scipy.ndimage.interpolation.rotate(i, v, reshape=False)
def img_scale(i, v):
    return np.clip(i * v, 0, 255)





