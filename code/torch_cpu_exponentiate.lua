x = torch.rand(1,10*30*768)

iters = 1000000

for i = 1, iters do
	y = torch.exp(x)
end

time = os.clock()
print("Time elapsed: ", time)
