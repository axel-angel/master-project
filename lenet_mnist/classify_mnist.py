import sys
import caffe
import matplotlib
import numpy as np
import lmdb

MODEL_FILE = 'lenet.prototxt'
PRETRAINED = 'snapshots/lenet_mnist_v2_iter_10000.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()
db_path = 'examples/mnist/mnist_test_lmdb'
lmdb_env = lmdb.open(db_path)
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
    print "\r", out['prob'].tolist()
    predicted_label = out['prob'][0].argmax(axis=0)
    if label == predicted_label[0][0]:
        correct = correct + 1
    else:
        print("\rError: expected %i but predicted %i" \
                % (label, predicted_label[0][0]))

    sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
    sys.stdout.flush()

print(str(correct) + " out of " + str(count) + " were classified correctly")
