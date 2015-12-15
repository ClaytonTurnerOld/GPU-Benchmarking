require 'cltorch'
require 'math'

x = torch.Tensor(1,10*30*768):uniform():cl()
y = torch.Tensor(1,10*30*768):zero():cl()

iters = 1000000

--begin = os.time()
for i=1,iters do
	y:apply("x = exp(x)")
	--x:map(y, "x = exp(x)")
end
--stop = os.time()

-- os.time() has second resolution - garbage for us
-- So we're going to use os.clock() even though it includes the overhead
--print("Time elapsed: ",os.difftime(stop,begin))
print("Time elapsed: ", os.clock())
