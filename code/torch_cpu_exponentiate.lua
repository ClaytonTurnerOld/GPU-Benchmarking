x = torch.rand(1,10*30*768)

for i = 1, 1000 do
	y = torch.exp(x)
end

time = os.clock()
print("Time elapsed: ", time)
