torch.setdefaulttensortype('torch.FloatTensor')
model = ""

function loadNeuralNetwork(path)
    print (path)
    print ("Loaded Neural Network -- Success")
    model = torch.load(path)

    print ("Model Architecture --\n")
    print (model)
    print ("---------------------\n")
end

function classifyExample(tensorInput)
    v = model(tensorInput)
	print(v[1])
	return v[1]
end