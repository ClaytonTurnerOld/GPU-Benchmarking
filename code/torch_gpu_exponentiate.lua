require 'cltorch'
require 'math'

x = torch.Tensor(1,10*30*768):uniform():cl()
--x = torch.Tensor(1,10):uniform():cl()
--print(x)
x:apply("x = exp(x)")
--print(x)

print("Time elapsed: ",os.clock())
