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
        self.isize = (28, 28)

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

        self.graphicsView_3 = QLabel(self.verticalLayoutWidget)

        self.graphicsView_4 = QLabel(self.verticalLayoutWidget)

        self.verticalLayout_3 = QVBoxLayout()

        self.addSliders()

        self.graphicsView_2 = QLabel(self.verticalLayoutWidget)

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout.addWidget(self.graphicsView_3)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        for s in self.sliders:
            self.verticalLayout_3.addWidget(s)
        self.horizontalLayout.addWidget(self.graphicsView_4)
        self.verticalLayout_2.addWidget(self.graphicsView_2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.loadImg(fpath)
        self.computeDisplays(self.imgnp)
        plot_tnse(self.graphicsView_2, pts, labels) # FIXME: slow (once)


    def computeDisplays(self, imgnp):
        # input display
        print "Plot input"
        qimg = QImage(imgnp.data, self.isize[0], self.isize[1],
                QImage.Format_Indexed8)
        pxmap = QPixmap.fromImage(qimg)
        self.graphicsView_3.setPixmap(pxmap.scaled(QtCore.QSize(28*5,28*5)))

        # probas display
        print "Forward network"
        res = net.forward_all(data=np.array([[ imgnp ]]))
        probas = res['prob'][0].flatten().tolist()
        print "probas:", probas
        print "Plot probas"
        plot_probas(self.graphicsView_4, probas)

        # t-SNE display
        print "Plot t-SNE"
        #plot_tnse(self.graphicsView_2, pts, labels) # FIXME: slow (once)

    def addSliders(self):
        self.sliders = []
        self.sfn = {
            'shift_x': lambda i,v: scipy.ndimage.interpolation.shift(i, (v,0)),
            'shift_y': lambda i,v: scipy.ndimage.interpolation.shift(i, (0,v)),
            'rotation': lambda i,v: scipy.ndimage.interpolation.rotate(i, v*3.6, reshape=False),
        }
        self.svalues = { k:0 for k in self.sfn.keys() }

        for k, sfn in self.sfn.iteritems():
            s = QSlider(self.verticalLayoutWidget)
            s.setOrientation(QtCore.Qt.Horizontal)
            s.setSliderPosition(50)
            self.sliders.append(s)
            fn = lambda v,k=k,sfn=sfn: self.onSlider(k, sfn, v - 50)
            QtCore.QObject.connect(s, QtCore.SIGNAL('valueChanged(int)'), fn)

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

# Source: http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image
def matplot2np(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels
        and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    return buf

def plot_tnse(widget, pts, labels):
    colors = labels
    fig = plt.figure(figsize=(6,4))
    plot = fig.add_subplot(1, 1, 1)
    plot.scatter(pts[:,0], pts[:,1], s=25, c=colors, cmap='bwr')
    plot.axis('off')
    plotnp = matplot2np(fig)
    isizex, isizey = plotnp.shape[0:2]

    qimg = QImage(plotnp.data, isizex, isizey, QImage.Format_ARGB32)
    pxmap = QPixmap.fromImage(qimg)
    widget.setPixmap(pxmap)
    plt.close()

def plot_probas(widget, probas):
    fig = plt.figure(figsize=(2,4))
    plot = fig.add_subplot(1, 1, 1)
    plot.barh(range(10), probas, height=0.5, align='center')
    plt.xlim(0, 1)
    plt.ylim(0, 10)
    plt.grid(True)
    plotnp = matplot2np(fig)
    isizex, isizey = plotnp.shape[0:2]

    qimg = QImage(plotnp.data, isizex, isizey, QImage.Format_ARGB32)
    pxmap = QPixmap.fromImage(qimg)
    widget.setPixmap(pxmap)
    plt.close()

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
