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
    def setupUi(self, MainWindow):
        MainWindow.resize(800, 600)

        self.centralwidget = QWidget(MainWindow)

        self.verticalLayoutWidget = QWidget(self.centralwidget)

        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setMargin(15)

        self.horizontalLayout = QHBoxLayout()

        self.graphicsView_3 = QLabel(self.verticalLayoutWidget)
        pxmap = QPixmap(fpath) \
                .scaled(QtCore.QSize(28,28)) \
                .scaled(QtCore.QSize(28*5,28*5))
        self.graphicsView_3.setPixmap(pxmap)

        imgnp = scipy.ndimage.imread(fpath, flatten=True)
        imgnp = scipy.misc.imresize(imgnp, size=(28,28))
        res = net.forward_all(data=np.array([[ imgnp ]]))
        print "res:", np.argmax(res['prob'][0])
        self.graphicsView_4 = QLabel(self.verticalLayoutWidget)
        plot_probas(self.graphicsView_4, res['prob'][0].tolist())

        self.verticalLayout_3 = QVBoxLayout()

        self.sliders = []
        for i in range(3):
            s = QSlider(self.verticalLayoutWidget)
            s.setOrientation(QtCore.Qt.Horizontal)
            self.sliders.append(s)
            fn = lambda v,i=i: self.onSlide(i, v)
            QtCore.QObject.connect(s, QtCore.SIGNAL('valueChanged(int)'), fn)

        self.graphicsView_2 = QLabel(self.verticalLayoutWidget)
        plot_tnse(self.graphicsView_2, pts, labels)

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

    def onSlide(self, i, value):
        print "slider:", i, value

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

def plot_probas(widget, probas):
    fig = plt.figure(figsize=(2,4))
    plot = fig.add_subplot(1, 1, 1)
    print "probas:", probas
    plot.hist(probas, len(probas), histtype='stepfilled', orientation='horizontal')
    plotnp = matplot2np(fig)
    isizex, isizey = plotnp.shape[0:2]

    qimg = QImage(plotnp.data, isizex, isizey, QImage.Format_ARGB32)
    pxmap = QPixmap.fromImage(qimg)
    widget.setPixmap(pxmap)

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
