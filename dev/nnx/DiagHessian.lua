
-- Module
function nn.Module.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or diagHessianOutput
   return self.diagHessianInput
end

function nn.Module.accDiagHessianParameters(self, input, diagHessianOutput, scale)
end

function nn.Module.initDiagHessianParameters(self)
end

-- Criterion
function nn.Criterion.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or self.output.new()
   return self.diagHessianInput
end

 -- MSECriterion
function nn.MSECriterion.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or input.new()
   self.diagHessianInput:resizeAs(input):fill(1)
   return self.diagHessianInput
end

-- Linear
function nn.Linear.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or self.output.new()
   self.weightSq = self.weightSq or self.output.new():resizeAs(self.weight)
   self.weightSq:copy(self.weight):cmul(self.weightSq)
   if input:dim() == 1 then
      self.diagHessianInput:resizeAs(input)
      self.diagHessianInput:addmv(0, 1, self.weightSq:t(), diagHessianOutput)
   elseif input:dim() == 2 then
      self.diagHessianInput:resizeAs(input)
      self.diagHessianInput:addmm(0, 1, diagHessianOutput, self.weightSq)
   end
   return self.diagHessianInput
end

function nn.Linear.initDiagHessianParameters(self)
   self.diagHessianWeight = self.diagHessianWeight or self.output.new():resizeAs(self.weight)
   self.diagHessianBias = self.diagHessianBias or self.output.new():resizeAs(self.bias)
end

function nn.Linear.accDiagHessianParameters(self, input, diagHessianOutput, scale)
   scale = scale or 1
   self.inputSq = self.inputSq or self.output.new()
   self.inputSq:resizeAs(input):copy(input):cmul(self.inputSq)
   if input:dim() == 1 then
      self.diagHessianWeight:addr(scale, diagHessianOutput, self.inputSq)
      self.diagHessianBias:add(scale, diagHessianOutput)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      self.diagHessianWeight:addmm(scale, diagHessianOutput:t(), self.inputSq)
      self.diagHessianBias:addmv(scale, diagHessianOutput:t(), self.output.new(nframe):fill(1))
   end
end

-- Tanh
function nn.Tanh.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or self.output.new()
   self.derivativeSq = self.derivativeSq or self.output.new()
   self.derivativeSq:resizeAs(self.output):copy(self.output):cmul(self.output):mul(-1):add(1)
   self.derivativeSq:cmul(self.derivativeSq)
   self.diagHessianInput:resizeAs(input):copy(diagHessianOutput):cmul(self.derivativeSq)
   return self.diagHessianInput
end

-- Sequential
function nn.Sequential.updateDiagHessianInput(self, input, diagHessianOutput)
   local currentDiagHessianOutput = diagHessianOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentDiagHessianOutput = currentModule:updateDiagHessianInput(previousModule.output, currentDiagHessianOutput)
      currentModule = previousModule
   end
   currentDiagHessianOutput = currentModule:updateDiagHessianInput(input, currentDiagHessianOutput)
   self.diagHessianInput = currentDiagHessianOutput
   return currentDiagHessianOutput
end

function nn.Sequential.initDiagHessianParameters(self)
   for i=1,#self.modules do
      self.modules[i]:initDiagHessianParameters()
   end
end

function nn.Sequential.accDiagHessianParameters(self, input, diagHessianOutput, scale)
   scale = scale or 1
   local currentDiagHessianOutput = diagHessianOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accDiagHessianParameters(previousModule.output, currentDiagHessianOutput, scale)
      currentDiagHessianOutput = currentModule.diagHessianInput
      currentModule = previousModule
   end
   currentModule:accDiagHessianParameters(input, currentDiagHessianOutput, scale)
end
