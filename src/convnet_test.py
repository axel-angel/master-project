import sys
import caffe
import numpy as np
import lmdb
import argparse
from collections import defaultdict

def lmdb_reader(fpath):
    lmdb_env = lmdb.open(args.lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, image, label)

def npz_reader(fpath):
    npz = np.load(fpath)

    xs = npz['arr_0']
    ls = npz['arr_1']

    for i, (x, l) in enumerate(np.array([ xs, ls ]).T):
        yield (i, x.reshape(1, *x.shape), l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lmdb', type=str, default=None)
    group.add_argument('--npz', type=str, default=None)
    args = parser.parse_args()

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()
    print "args", vars(args)
    if args.lmdb != None:
        reader = lmdb_reader(args.lmdb)
    if args.npz != None:
        reader = npz_reader(args.npz)

    for i, image, label in reader:
        out = net.forward_all(data=np.asarray([image]))
        plabel = int(out['prob'][0].argmax(axis=0))

        count += 1
        iscorrect = label == plabel
        correct += (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if not iscorrect:
            print("\rError: i=%s, expected %i but predicted %i" \
                    % (i, label, plabel))

        sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        sys.stdout.flush()

    print(", %i/%i corrects" % (correct, count))

    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])
