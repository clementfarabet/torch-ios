local KLDivCriterion, parent = torch.class('nn.KLDivCriterion', 'nn.Criterion')

function KLDivCriterion:__init(opts)
   parent.__init(self)
   -- user options
   opts = opts or {}
   self.inputIsProbability = opts.inputIsProbability or false
   self.targetIsProbability = opts.targetIsProbability
   if self.targetIsProbability == nil then self.targetIsProbability = true end
   -- internal
   self.targetSoftMax = nn.SoftMax()
   self.inputSoftMax = nn.SoftMax()
   self.gradProbInput = torch.Tensor()
end

function KLDivCriterion:normalize(input, target)
   -- normalize target
   if not self.targetIsProbability then
      self.probTarget = self.targetSoftMax:updateOutput(target)
   else
      self.probTarget = target
   end

   -- normalize input
   if not self.inputIsProbability then
      self.probInput = self.inputSoftMax:updateOutput(input)
   else
      self.probInput = input
   end
end

function KLDivCriterion:denormalize(input)
   -- denormalize gradients
   if not self.inputIsProbability then
      self.gradInput = self.inputSoftMax:updateGradInput(input, self.gradProbInput)
   else
      self.gradInput = self.gradProbInput
   end
end

function KLDivCriterion:updateOutput(input, target)
   self:normalize(input, target)
   self.output = 0
   for i = 1,input:size(1) do
      local acc = 0
      if self.probTarget[i] > 0 then
         acc = self.probTarget[i] * math.log(self.probTarget[i] / math.max(self.probInput[i],1e-9))
      end
      self.output = self.output + acc
   end
   return self.output
end

function KLDivCriterion:updateGradInput(input, target)
   self:normalize(input, target)
   self.gradProbInput:resizeAs(input)
   for i = 1,input:size(1) do
      self.gradProbInput[i] = - self.probTarget[i] / math.max(self.probInput[i],1e-9)
   end
   self:denormalize(input)
   return self.gradInput
end
