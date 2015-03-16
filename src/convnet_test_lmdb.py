import sys
import caffe
import matplotlib
import numpy as np
import lmdb
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lmdb', type=str, required=True)
    args = parser.parse_args()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()
    lmdb_env = lmdb.open(args.lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    for key, value in lmdb_cursor:
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        out = net.forward_all(data=np.asarray([image]))
        predicted_label = out['prob'][0].argmax(axis=0)
        if label == predicted_label[0][0]:
            correct = correct + 1
        else:
            print("\rError: expected %i but predicted %i" \
                    % (label, predicted_label[0][0]))

        sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        sys.stdout.flush()

    print(str(correct) + " out of " + str(count) + " were classified correctly")
