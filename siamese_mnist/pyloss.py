import caffe
import numpy as np

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
        self.zeros = np.zeros(bottom[0].num)
        self.m = 1.0
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        GW1 = bottom[0].data
        GW2 = bottom[1].data
        Y = bottom[2].data
        loss = 0.0
        self.diff = GW1 - GW2
        self.dist_sq = np.sum(self.diff**2, axis=1)
        losses = Y * self.dist_sq \
           + (1-Y) * np.max([self.zeros, self.m - self.dist_sq], axis=0)
        loss = np.sum(losses)
        top[0].data[0] = loss / 2.0 / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        Y = bottom[2].data
        disClose = np.where(self.m - self.dist_sq > 0.0, 1.0, 0.0)
        for i, sign in enumerate([ +1, -1 ]):
            if propagate_down[i]:
                alphas = np.where(Y > 0, +1.0, -1.0) * sign * top[0].diff[0] / bottom[i].num
                facts = ((1-Y) * disClose + Y) * alphas
                bottom[i].diff[...] = np.array([facts, facts]).T * self.diff

class OwnDoubleContrastiveLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff1 = np.zeros(bottom[0].num, dtype=np.float32)
        self.diff2 = np.zeros(bottom[0].num, dtype=np.float32)
        self.dist_sq1 = np.zeros(bottom[0].num, dtype=np.float32)
        self.dist_sq2 = np.zeros(bottom[0].num, dtype=np.float32)
        self.zeros = np.zeros(bottom[0].num)
        self.m = 1.0
        self.n = 1.0
        self.C = 1.0
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        GW1f = bottom[0].data[:,0]
        GW2f = bottom[1].data[:,0]
        GW1g = bottom[0].data[:,1]
        GW2g = bottom[1].data[:,1]
        Y = np.right_shift(bottom[2].data.astype(int), 0) & 1 # digit label
        Z = np.right_shift(bottom[2].data.astype(int), 1) & 1 # transfo label
        loss = 0.0
        self.diff1 = GW1f - GW2f
        self.diff2 = GW1g - GW2g
        self.dist_sq1 = self.diff1**2
        self.dist_sq2 = self.diff2**2
        losses1 = Y * self.dist_sq1 \
           + (1-Y) * np.max([self.zeros, self.m - self.dist_sq1], axis=0)
        losses2 = Z * self.dist_sq2 \
           + (1-Z) * np.max([self.zeros, self.n - self.dist_sq2], axis=0)
        loss = np.sum(losses1) + self.C * np.sum(losses2)
        top[0].data[0] = loss / 2.0 / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        Y = np.right_shift(bottom[2].data.astype(int), 0) & 1 # digit label
        Z = np.right_shift(bottom[2].data.astype(int), 1) & 1 # transfo label
        disClose1 = np.where(self.m - self.dist_sq1 > 0.0, 1.0, 0.0)
        disClose2 = np.where(self.n - self.dist_sq2 > 0.0, 1.0, 0.0)
        for i, sign in enumerate([ +1, -1 ]):
            if propagate_down[i]:
                alphas1 = np.where(Y > 0, +1.0, -1.0) * sign * top[0].diff[0] / bottom[i].num
                alphas2 = np.where(Z > 0, +1.0, -1.0) * sign * top[0].diff[0] / bottom[i].num
                facts1 = ((1-Y) * disClose1 + Y) * alphas1
                facts2 = ((1-Z) * disClose2 + Z) * alphas2
                bottom[i].diff[:,0] = facts1 * self.diff1
                bottom[i].diff[:,1] = facts2 * self.diff2 * self.C

class OwnDouble2DContrastiveLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff1 = np.zeros(bottom[0].num, dtype=np.float32)
        self.diff2 = np.zeros(bottom[0].num, dtype=np.float32)
        self.dist_sq1 = np.zeros(bottom[0].num, dtype=np.float32)
        self.dist_sq2 = np.zeros(bottom[0].num, dtype=np.float32)
        self.zeros = np.zeros(bottom[0].num)
        self.m = 1.0
        self.n = 1.0
        self.C = 1.0
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        GW1f = bottom[0].data[:,0:2] # 2D
        GW2f = bottom[1].data[:,0:2] # 2D
        GW1g = bottom[0].data[:,2] # 1D
        GW2g = bottom[1].data[:,2] # 1D
        Y = np.right_shift(bottom[2].data.astype(int), 0) & 1 # digit label
        Z = np.right_shift(bottom[2].data.astype(int), 1) & 1 # transfo label
        loss = 0.0
        self.diff1 = GW1f - GW2f
        self.diff2 = GW1g - GW2g
        self.dist_sq1 = np.sum(self.diff1**2, axis=1) # 2D -> 1D
        self.dist_sq2 = self.diff2**2
        losses1 = Y * self.dist_sq1 \
           + (1-Y) * np.max([self.zeros, self.m - self.dist_sq1], axis=0)
        losses2 = Z * self.dist_sq2 \
           + (1-Z) * np.max([self.zeros, self.n - self.dist_sq2], axis=0)
        loss = np.sum(losses1) + self.C * np.sum(losses2)
        top[0].data[0] = loss / 2.0 / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        Y = np.right_shift(bottom[2].data.astype(int), 0) & 1 # digit label
        Z = np.right_shift(bottom[2].data.astype(int), 1) & 1 # transfo label
        disClose1 = np.where(self.m - self.dist_sq1 > 0.0, 1.0, 0.0)
        disClose2 = np.where(self.n - self.dist_sq2 > 0.0, 1.0, 0.0)
        for i, sign in enumerate([ +1, -1 ]):
            if propagate_down[i]:
                alphas1 = np.where(Y > 0, +1.0, -1.0) * sign * top[0].diff[0] / bottom[i].num
                alphas2 = np.where(Z > 0, +1.0, -1.0) * sign * top[0].diff[0] / bottom[i].num
                facts1 = (((1-Y) * disClose1 + Y) * alphas1).reshape(-1, 1)
                facts2 =  ((1-Z) * disClose2 + Z) * alphas2
                bottom[i].diff[:,0:2] = facts1 * self.diff1
                bottom[i].diff[:,2]   = facts2 * self.diff2 * self.C

class OwnAlignerLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff1 = np.zeros((bottom[0].num, 2), dtype=np.float32)
        self.diff2 = np.zeros((bottom[0].num, 2), dtype=np.float32)
        self.dist_sq1 = np.zeros(bottom[0].num, dtype=np.float32)
        self.dist_sq2 = np.zeros(bottom[0].num, dtype=np.float32)
        self.zeros = np.zeros(bottom[0].num)
        self.m = 2.0
        self.d = np.array([ 1.0, 0.0 ]) # FIXME: 2D
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        GW1 = bottom[0].data
        GW2 = bottom[1].data
        Y = bottom[2].data
        loss = 0.0
        Q = (GW1 + GW2) / 2.0
        P1B = Q + self.d
        self.diff1 = P1B - GW1
        self.diff2 = GW2 - GW1
        self.dist_sq1 = np.sum(self.diff1**2, axis=1)
        self.dist_sq2 = np.sum(self.diff2**2, axis=1)
        losses = Y * self.dist_sq1 \
           + (1-Y) * np.max([self.zeros, self.m - self.dist_sq2], axis=0)
        loss = np.sum(losses)
        top[0].data[0] = loss / 2.0 / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        Y = bottom[2].data
        disClose = np.where(self.m - self.dist_sq2 > 0.0, 1.0, 0.0)
        for i, sign in enumerate([ +1, -1 ]):
            if propagate_down[i]:
                alphas = np.where(Y > 0, +1.0, -1.0) * sign * top[0].diff[0] / bottom[i].num
                facts1 = np.repeat([Y * alphas], 2, axis=0).T
                facts2 = np.repeat([(1-Y) * disClose * alphas], 2, axis=0).T
                bottom[i].diff[...] = facts1 * self.diff1 \
                                    + facts2 * self.diff2
