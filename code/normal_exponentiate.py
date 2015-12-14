# We are leaving numpy code in here so we have consistent results considering the values of the elements in arrays
# We convert back to Python native lists before timing to avoid timing bias

import numpy
from timeit import default_timer as timer # Since we need to use Python 2.7, this is a good alternative to Perfcounter

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

# Keep consistent results with random seed
rng = numpy.random.RandomState(42)

# float32 so we have a fair comparison to the Theano times
# rng.rand(z) creates an array of size z
x = numpy.asarray(rng.rand(vlen), numpy.float32)
x = x.tolist()

def exp(exp_list):
    for i in xrange(len(exp_list)):
        exp_list[i] = exp_list[i] ** 2
    return exp_list

begin = timer()
for i in xrange(iters):
        r = exp(x)
end = timer()
print("Looping %d times took %f seconds" % (iters, end - begin))
# We're not printing the result because python doesn't, by default, have the pretty printing seen in Theano and Numpy by default
#print("Result is %s" % (r,))
