# Adapted from deeplearning.net

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
from timeit import default_timer as timer # Since we need to use Python 2.7, this is a good alternative to Perfcounter

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

# Keep consistent results with random seed
rng = numpy.random.RandomState(42)

# Use Theano's shared variables in order to distribute computation to the GPU
# floatX == float32 must hold True for GPU to work - command line specification
# rng.rand(z) creates an array of size z
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))

# Function syntax:
#   arg[0] = parameters to be passed in
#   arg[1] = the function to be performed
# We pass our shared array as part of the function definition
f = function([], T.exp(x))

begin = timer()
for i in xrange(iters):
        r = f()
end = timer()
print("Looping %d times took %f seconds" % (iters, end - begin))
print("Result is %s" % (r,))

# Let's make sure we used the correct processing unit by printing and comparing to what we expect
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
