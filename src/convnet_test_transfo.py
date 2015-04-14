import sys
import caffe
import numpy as np
import lmdb
import argparse
from collections import defaultdict
from utils import lmdb_reader, npz_reader, parse_transfo
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--transfo', type=parse_transfo, action='append',
            default=[])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lmdb', type=str, default=None)
    group.add_argument('--npz', type=str, default=None)
    args = parser.parse_args()

    print "Load model"
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_cpu()
    print "args", vars(args)
    if args.lmdb != None:
        reader = lmdb_reader(args.lmdb)
    if args.npz != None:
        reader = npz_reader(args.npz)

    trs = [ { 'f': getattr(utils, 'img_%s' % (tr)), 'name': tr,
              'steps': lambda: range(x, y, np.sign(y-x)),
              'maxf': np.max if (np.sign(y-x) > 0) else np.min }
            for (tr,x,y) in args.transfo ]
    trs_len = len(trs)
    print "Transformations: %s" % ("\n\t".join(map(repr, trs)))

    count = 0
    count_all = 0
    correct = 0
    correct_vmax = defaultdict(int) # extreme disto value correctly classified
    labels_set = set()

    print "Test network against transformations"
    for i, image, label in reader:
        trf = [ (t['f'], v, t['name'], t['maxf'])
                for t in trs for v in t['steps']() ]
        for j, (f, v, name, maxf) in enumerate(trf):
            out = net.forward_all(data=np.asarray([image]))
            plabel = int(out['prob'][0].argmax(axis=0))

            count_all += 1
            iscorrect = label == plabel
            if iscorrect:
                correct += 1
            else:
                vmax = maxf(v, correct_vmax[(j, name)])
                correct_vmax[(label, j, name)] = vmax
                break

        count += 1

        sys.stdout.write("\rRunning: %i" % (count))
        sys.stdout.flush()

    print ""
    print "Extremum correct classification:"
    print "(l, j , tr) | count"
    for ((l, j, name), v) in correct_vmax.iteritems():
        print "(%i, %i , %s) | %i" % (l, j, name, v)
