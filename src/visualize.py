#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtCore
from PyQt4.QtGui import *
import numpy as np
from functools import partial
import caffe
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.misc
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.manifold.t_sne import _gradient_descent as tsne_gd, _joint_probabilities as tsne_joint_probabilities, _kl_divergence as tsne_kl_divergence
from sklearn.metrics.pairwise import pairwise_distances
import utils

try:
    _encoding = QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def __init__(self):
        self.fpath = None
        self.imgnp = None

    def loadImg(self, fpath):
        imgnp = scipy.ndimage.imread(fpath, flatten=True)
        imgnp = scipy.misc.imresize(imgnp, size=(28,28))
        self.fpath = fpath
        self.imgnp = imgnp

    def setupUi(self, MainWindow):
        MainWindow.resize(800, 600)

        self.centralwidget = QWidget(MainWindow)

        self.verticalLayoutWidget = QWidget(self.centralwidget)

        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setMargin(15)

        self.horizontalLayout = QHBoxLayout()

        self.figure_0 = Figure((2, 2))
        self.figure_2 = Figure((6, 4))
        self.figure_4 = Figure((2, 4))
        self.figure_5 = Figure((6, 4))
        self.figureCanvas_0 = FigureCanvas(self.figure_0)
        self.figureCanvas_4 = FigureCanvas(self.figure_4)
        self.figureCanvas_5 = FigureCanvas(self.figure_5)

        self.verticalLayout_3 = QVBoxLayout()

        self.addSliders()

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout.addWidget(self.figureCanvas_0)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout.addWidget(self.figureCanvas_4)
        self.verticalLayout_2.addWidget(self.figureCanvas_5)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.loadImg(fpath)
        self.computeDisplays(self.imgnp)


    def computeDisplays(self, imgnp):
        # input display
        print "Plot input"
        self.plot_input(imgnp)

        # probas display
        print "Forward network"
        res = net.forward_all(data=np.array([[ imgnp ]]), blobs=['conv1'])
        probas = res['prob'][0].flatten().tolist()
        print "probas:", probas
        print "Plot probas"
        self.plot_probas(probas)

        # features display
        print "Plot conv1 features"
        fsconv1 = res['conv1'][0]
        self.plot_features(fsconv1)

    def addSliders(self):
        self.sliders = []
        self.sfn = {
            'shift_x': lambda i,v: utils.img_shift_x(i, v*28),
            'shift_y': lambda i,v: utils.img_shift_y(i, v*28),
            'blur': lambda i,v: utils.img_blur(i, v*10),
            'rotation': lambda i,v: utils.img_rotate(i, v*180.),
            'scale': lambda i,v: utils.img_scale(i, 1+v),
            'sindisp_x': lambda i,v: utils.img_sindisp_x(i, v*10),
            'sindisp_y': lambda i,v: utils.img_sindisp_y(i, v*10),
        }
        self.svalues = { k:0 for k in self.sfn.keys() }

        for k, sfn in self.sfn.iteritems():
            s = QSlider(self.verticalLayoutWidget)
            s.setOrientation(QtCore.Qt.Horizontal)
            s.setSliderPosition(50)
            self.sliders.append(s)
            fn = lambda v,k=k,sfn=sfn: self.onSlider(k, sfn, v/50. - 1.)
            QtCore.QObject.connect(s, QtCore.SIGNAL('valueChanged(int)'), fn)
            self.verticalLayout_3.addWidget(QLabel(k))
            self.verticalLayout_3.addWidget(s)

    def onSlider(self, k, sfn, value):
        print "slider:", k, sfn, value
        self.svalues[k] = value

        # apply transformations
        imgnp = self.imgnp
        for k, sfn in self.sfn.iteritems():
            imgnp = sfn(imgnp, self.svalues[k])

        print "Sliders:", self.svalues
        self.computeDisplays(imgnp)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))

    def plot_input(self, data):
        fig = self.figure_0
        canvas = self.figureCanvas_0

        fig.clf()
        plot = fig.add_subplot(1, 1, 1)
        plot.axis('off')
        plot.imshow(data, cmap='gray', interpolation='nearest',
                vmin=0, vmax=255)
        canvas.draw()

    def plot_probas(self, probas):
        fig = self.figure_4
        canvas = self.figureCanvas_4

        fig.clf()
        plot = fig.add_subplot(1, 1, 1)
        plot.barh(range(10), probas, height=0.5, align='center')
        plot.set_xlim(0, 1)
        plot.set_ylim(0, 10)
        plot.grid(True)
        canvas.draw()

    # inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb
    def plot_features(self, data, padsize=1, padval=0):
        fig = self.figure_5
        canvas = self.figureCanvas_5

        data -= data.min()
        data /= data.max() - data.min()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) \
                + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]) \
                   .transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        fig.clf()
        plot = fig.add_subplot(1, 1, 1)
        plot.axis('off')
        plot.imshow(data, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        canvas.draw()


class StartQT4(QMainWindow):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    fpath = args.image

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    app = QApplication(sys.argv)
    myapp = StartQT4()
    myapp.show()
    sys.exit(app.exec_())
