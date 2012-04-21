local Type, parent = torch.class('nn.Type', 'nn.Sequential')

function Type:__init(type)
   parent.__init(self)
   if not type:find('torch%..+Tensor') then
      type = 'torch.' .. type .. 'Tensor'
   end
   self.type = type
   self.defaulttype = torch.getdefaulttensortype()
   self.convert_input = nn.Copy(self.defaulttype, self.type)
   self.convert_gradOutput = nn.Copy(self.defaulttype, self.type)
   self.convert_output = nn.Copy(self.type, self.defaulttype)
   self.convert_gradInput = nn.Copy(self.type, self.defaulttype)
end

function Type:add(module)
   parent.add(self, module:type(self.type))
   return self
end

function Type:updateOutput(input)
   input = self.convert_input:updateOutput(input)
   local output = parent.updateOutput(self, input)
   self.output = self.convert_output:updateOutput(output)
   return self.output
end

function Type:updateGradInput(input, gradOutput)
   input = self.convert_input:updateOutput(input)
   gradOutput = self.convert_gradOutput:updateOutput(gradOutput)
   local gradInput = parent.updateGradInput(self, input, gradOutput)
   self.gradInput = self.convert_gradInput:updateOutput(gradInput)
   return self.gradInput
end

function Type:accGradParameters(input, gradOutput)
   input = self.convert_input:updateOutput(input)
   gradOutput = self.convert_gradOutput:updateOutput(gradOutput)
   parent.accGradParameters(self, input, gradOutput)
end

local Float, parent = torch.class('nn.Float', 'nn.Type')
function Float:__init()
   parent.__init(self, 'torch.FloatTensor')
end

local Double, parent = torch.class('nn.Double', 'nn.Type')
function Double:__init()
   parent.__init(self, 'torch.DoubleTensor')
end

local Cuda, parent = torch.class('nn.Cuda', 'nn.Type')
function Cuda:__init()
   parent.__init(self, 'torch.CudaTensor')
end
