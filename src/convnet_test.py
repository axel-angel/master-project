#!/usr/bin/python
# -*- encoding: utf8 -*-

import caffe
import numpy as np
import sys
from collections import defaultdict
import argparse

np.set_printoptions(suppress=True) # no sci notation

parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, help='deploy .prototxt', required=True)
parser.add_argument('--model', type=str, help='.caffemodel', required=True)
#parser.add_argument('--mean', type=str, help='mean .npy', default=None)
#parser.add_argument('--threshold', type=float, help='maxarg', default=0.5)
parser.add_argument('--npz', type=str,
        help='numpy arrays data + labels', required=True)
#parser.add_argument('--dimensions', type=int, nargs='+',
#        help='image dimension', required=True)

args = parser.parse_args()

caffe.set_mode_cpu()
#caffe.set_phase_test()

X = np.load(args.npz)
data = X['arr_0']
labels = X['arr_1']

labels_set = set()

assert data.shape[0] == labels.shape[0]
count = data.shape[0]

dims = (data.shape[1], data.shape[2])
m = caffe.Net(args.proto, args.model)

matrix = defaultdict(int) # (real,pred) -> int

for i in range(count):
    label = labels[i]
    out = m.forward_all(data=np.asarray([ np.array([ data[i] ]) ]))

    # our pred is the one above threshold else 0 (bias default)
    #maxpred = np.max(pred)
    #plabel = np.argmax(pred)
    plabel = int(np.argmax(out['prob'][0], axis=0))

    iscorrect = plabel == labels[i]
    matrix[(label, plabel)] += 1
    labels_set.update([label, plabel])

    if not iscorrect:
        print "\rError: expected %i, got %i, for %i (pred: %s)" \
                % (label, plabel, i, out['prob'][0].tolist())

    def cc_gen():
        for l in labels_set:
            crt = matrix[(l,l)]
            cnt = sum(x for (l2,pl), x in matrix.iteritems() if l2 == l)
            pre = sum(x for (l2,pl), x in matrix.iteritems() if pl == l)

            precis = 100. * crt / pre if pre != 0 else 0
            recall = 100. * crt / cnt if cnt != 0 else 0

            yield (l, cnt, precis, recall)

    cc_str = " | ".join("%s:%i %.1f%%/%.1f%%" % xs for xs in cc_gen())

    correct = sum(x for (l,pl), x in matrix.iteritems() if l == pl)
    count = sum(x for (l,pl), x in matrix.iteritems())

    sys.stdout.write("\rAccuracy: %.3f%% (%i | %s)" \
            % (100.*correct/count, count, cc_str))
    sys.stdout.flush()

print ""
print "Confusion matrix:"
print "(r , p) | count"
for l in labels_set:
    for pl in labels_set:
        print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])
