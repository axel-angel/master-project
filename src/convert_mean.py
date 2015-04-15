#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../caffe/python/')
import numpy as np
import sys
import caffe

if len(sys.argv) != 3:
    print "Usage: python convert_protomean.py proto.mean out.npy"
    sys.exit()

print "Convert %s -> %s" % (sys.argv[1], sys.argv[2])

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(sys.argv[1], 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save(sys.argv[2], out)
