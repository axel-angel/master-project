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

        self.inputView = QLabel(self.verticalLayoutWidget)

        self.figure_2 = Figure((6, 4))
        self.figure_4 = Figure((2, 4))
        self.figure_5 = Figure((6, 4))
        self.figureCanvas_2 = FigureCanvas(self.figure_2)
        self.figureCanvas_4 = FigureCanvas(self.figure_4)
        self.figureCanvas_5 = FigureCanvas(self.figure_5)

        self.verticalLayout_3 = QVBoxLayout()

        self.addSliders()

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout.addWidget(self.inputView)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout.addWidget(self.figureCanvas_4)
        #self.verticalLayout_2.addWidget(self.figureCanvas_2) # FIXME
        self.verticalLayout_2.addWidget(self.figureCanvas_5)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.loadImg(fpath)
        self.computeDisplays(self.imgnp)


    def computeDisplays(self, imgnp):
        # input display
        print "Plot input"
        qimg = QImage(imgnp.data, imgnp.shape[0], imgnp.shape[1],
                QImage.Format_Indexed8)
        pxmap = QPixmap.fromImage(qimg)
        self.inputView.setPixmap(pxmap.scaled(QtCore.QSize(28*5,28*5)))

        # probas display
        print "Forward network"
        res = net.forward_all(data=np.array([[ imgnp ]]), blobs=['conv1'])
        probas = res['prob'][0].flatten().tolist()
        print "probas:", probas
        print "Plot probas"
        plot_probas(self.figure_4, self.figureCanvas_4, probas)

        # t-SNE display
        print "Plot t-SNE"
        #plot_tnse(self.figure_2, self.figureCanvas_2, pts, labels) # FIXME: slow (once)

        # features display
        print "Plot conv1 features"
        fsconv1 = res['conv1'][0]
        plot_features(self.figure_5, self.figureCanvas_5, fsconv1)

    def addSliders(self):
        self.sliders = []
        self.sfn = {
            'shift_x': lambda i,v: scipy.ndimage.interpolation.shift(i, (v*28,0)),
            'shift_y': lambda i,v: scipy.ndimage.interpolation.shift(i, (0,v*28)),
            'blur': lambda i,v: scipy.ndimage.filters.gaussian_filter(i, v*10),
            'rotation': lambda i,v: scipy.ndimage.interpolation.rotate(i, v*180., reshape=False),
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

        self.computeDisplays(imgnp)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))

class StartQT4(QMainWindow):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

def plot_tnse(fig, canvas, pts, labels):
    colors = labels
    fig.clf()
    plot = fig.add_subplot(1, 1, 1)
    plot.scatter(pts[:,0], pts[:,1], s=25, c=colors, cmap='bwr')
    plot.axis('off')
    canvas.draw()

def plot_probas(fig, canvas, probas):
    fig.clf()
    plot = fig.add_subplot(1, 1, 1)
    plot.barh(range(10), probas, height=0.5, align='center')
    plot.set_xlim(0, 1)
    plot.set_ylim(0, 10)
    plot.grid(True)
    canvas.draw()

# inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb
def plot_features(fig, canvas, data, padsize=1, padval=0):
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

    fig.clf()
    plot = fig.add_subplot(1, 1, 1)
    plot.axis('off')
    plot.imshow(data, cmap='gray')
    canvas.draw()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--tsne-npz', type=str, required=True)
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    fpath = args.image

    X = np.load(args.tsne_npz)
    tsne = X['tsne']
    pca = X['pca']
    pts = X['pts']
    labels = X['labels']

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()

    app = QApplication(sys.argv)
    myapp = StartQT4()
    myapp.show()
    sys.exit(app.exec_())
