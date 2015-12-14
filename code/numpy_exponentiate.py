import numpy
from timeit import default_timer as timer # Since we need to use Python 2.7, this is a good alternative to Perfcounter

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

# Keep consistent results with random seed
rng = numpy.random.RandomState(42)

# float32 so we have a fair comparison to the Theano times
# rng.rand(z) creates an array of size z
x = numpy.asarray(rng.rand(vlen), numpy.float32)

begin = timer()
for i in xrange(iters):
        r = numpy.exp(x)
end = timer()
print("Looping %d times took %f seconds" % (iters, end - begin))
print("Result is %s" % (r,))
