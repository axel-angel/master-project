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
        self.dist_sq = np.zeros(bottom[0].num, dtype=np.float32)
        self.m = 1.0
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        GW1 = bottom[0].data
        GW2 = bottom[1].data
        Y = bottom[2].data
        loss = 0.0
        self.diff = GW1 - GW2
        for i in xrange(bottom[0].num):
            dist_sq = np.dot(self.diff[i], self.diff[i])
            self.dist_sq[i] = dist_sq
            if Y[i]: # similar pairs
                loss += self.dist_sq[i]
            else: # dissimilar pair
                loss += max(0.0, self.m - self.dist_sq[i])
        top[0].data[0] = loss / 2.0 / bottom[0].num
        #print "locals", locals()
        #raise Exception("Stop")

    def backward(self, top, propagate_down, bottom):
        for i, sign in enumerate([ +1, -1 ]):
            if propagate_down[i]:
                alpha = sign * top[0].diff[0] / bottom[i].num
                for j in xrange(bottom[i].num):
                    if bottom[2].data[j]: # similar pairs
                        bottom[i].data[j,...] = self.diff[j] * +alpha
                    else: # dissimilar pairs
                        if self.m - self.dist_sq[j] > 0.0:
                            bottom[i].data[j,...] = self.diff[j] * -alpha
                        else:
                            bottom[i].data[j,...] = 0.0
