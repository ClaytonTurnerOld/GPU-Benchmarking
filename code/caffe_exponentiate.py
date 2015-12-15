from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from timeit import default_timer as timer
import numpy

# helper function for common structures
vlen = 10 * 30 * 768
rng = numpy.random.RandomState(42)
blob = caffe_pb2.BlobProto()
blob.data.extend(list(rng.rand(vlen)))
data = blob.data

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def euc_loss(x):
    return L.EuclideanLoss(x,x)

def caffenet(lmdb, batch_size=256, include_acc=False):
    euc1 = euc_loss(data)
    #loss = L.SoftmaxWithLoss(euc1, label)
    #return to_proto(loss)
    return None

def make_net():
    iters = 1000000
    begin = timer()
    for i in xrange(iters):
        with open('../data/train.prototxt', 'w') as f:
            print(caffenet('/path/to/caffe-train-lmdb'), file=f)
        with open('../data/test.prototxt', 'w') as f:
            print(caffenet('/path/to/caffe-val-lmdb', batch_size=50, include_acc=True), file=f)
    end = timer()
    print("Looping %d times took %f seconds" % (iters, end - begin))

if __name__ == '__main__':
    make_net()
