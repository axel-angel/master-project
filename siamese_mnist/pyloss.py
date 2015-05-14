import caffe
import numpy as np

inf = float('inf') # +infinity

class OwnContrastiveLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros(bottom[0].num, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        GW1 = bottom[0].data
        GW2 = bottom[1].data
        Y = bottom[2].data
        DW2 = np.sum( (GW1 - GW2)**2 , axis=1)
        DW = np.sqrt( DW2 )
        m = 1.0 # dissimilar margin
        mdiffmax = np.clip(m - DW2, 0, inf)
        loss = np.sum( np.multiply(Y, DW2) + np.multiply((1-Y), mdiffmax) )
        top[0].data[...] = loss / 2.0 / bottom[0].num
        self.diff[...] = np.multiply(Y, DW) - np.multiply((1-Y), np.clip( m - DW, 0, inf) )
        print "locals", locals()
        raise Exception("Stop")

    def backward(self, top, propagate_down, bottom):
        for i, sign in enumerate([ +1, -1 ]):
            if propagate_down[i]:
                Ndiffs = np.repeat([ self.diff ], bottom[i].channels, axis=0).T
                bottom[i].diff[...] = sign * Ndiffs / bottom[i].num
