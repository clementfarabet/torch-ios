require 'torch'

mlp = nn.Sequential();
inputs = 2; outputs = 1; HUs = 20;
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))

criterion = nn.MSECriterion()

for i = 1,2500 do
  local input= torch.randn(2); 
  local output= torch.Tensor(1);
  if input[1]*input[2] > 0 then
    output[1] = -1
  else
    output[1] = 1
  end

  criterion:forward(mlp:forward(input), output)
  mlp:zeroGradParameters()
  mlp:backward(input, criterion:backward(mlp.output, output))
  mlp:updateParameters(0.01)
end

function classifyExample(tensorInput)
  v = mlp:forward(tensorInput)
	print(v[1])
	return v[1]
end