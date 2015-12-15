import pyopencl as cl
import numpy
import sys
from timeit import default_timer as timer
 
class CL(object):
    def __init__(self, size=10*30*768):
        self.size = size
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.iters = 1000000
 
    def load_program(self):
        # Using exponentiate instead of exp to avoid overloading/overriding
        fstr="""
        __kernel void exponentiate(__global float* a, __global float* c)
        {
            unsigned int i = get_global_id(0);
 
           c[i] = a[i] * a[i];
        }
         """
        self.program = cl.Program(self.ctx, fstr).build()
 
    def popCorn(self):
        mf = cl.mem_flags
 	
	# We use float32s here to have a fair comparison to the CPU
        self.a = numpy.array(range(self.size), dtype=numpy.float32)
 
        self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=self.a)
        self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes)
 
    def execute(self):
        begin = timer()
        for i in xrange(self.iters):
            self.program.exponentiate(self.queue, self.a.shape, None, self.a_buf, self.dest_buf)
        end = timer()
        c = numpy.empty_like(self.a)
        cl.enqueue_read_buffer(self.queue, self.dest_buf, c).wait()
        print("Looping %d times took %f seconds" % (self.iters, end - begin))
        print("Result is %s" % (c))
 
if __name__ == '__main__':
    matrixmul = CL(10000000)
    matrixmul.load_program()
    matrixmul.popCorn()
    matrixmul.execute()
