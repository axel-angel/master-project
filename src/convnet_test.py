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
parser.add_argument('--mean', type=str, help='mean .npy', required=True)
parser.add_argument('--path', type=str, help='directory', required=True)
parser.add_argument('--threshold', type=float, help='maxarg', default=0.5)
parser.add_argument('--filelist', type=str,
        help='file list with labels', required=True)
parser.add_argument('--dimensions', type=int, nargs='+',
        help='image dimension', required=True)

args = parser.parse_args()

caffe.set_mode_cpu()
caffe.set_phase_test()

dims = tuple(args.dimensions) + (3,)
m = caffe.Classifier(args.proto, args.model,
        image_dims=dims, channel_swap=(2,1,0), raw_scale=255,
        mean=np.load(args.mean))

matrix = defaultdict(int) # (real,pred) -> int
labels = set()

for line in open(args.filelist, 'r'):
    fpath, label = map(lambda (f,x): f(x), zip([str, int], line.split(' ', 1)))
    img = caffe.io.load_image(args.path + fpath)
    pred = m.predict([img])

    # our pred is the one above threshold else 0 (bias default)
    maxpred = np.max(pred)
    plabel = np.argmax(pred) if maxpred > args.threshold else 0

    iscorrect = plabel == label
    matrix[(label, plabel)] += 1
    labels.update([label, plabel])

    if not iscorrect:
        print "\rError: expected %i, got %i, for %s (pred: %s)" \
                % (label, plabel, fpath, pred)

    def cc_gen():
        for l in labels:
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
for l in labels:
    for pl in labels:
        print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])
