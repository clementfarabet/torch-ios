local SpatialClassNLLCriterion, parent = torch.class('nn.SpatialClassNLLCriterion', 'nn.ClassNLLCriterion')

function SpatialClassNLLCriterion:__init(...)
   parent.__init(self,...)

   xlua.unpack_class(self, {...},
      'nn.SpatialClassNLLCriterion',
      'A spatial extension of the NLLCriterion class.\n'
        ..' Provides a set of parameters to deal with spatial mini-batch training.',
      {arg='resampleTarget', type='number', help='ratio to resample target (target is a KxHxW tensor)', default=1},
      {arg='nbGradients', type='number', help='number of gradients to backpropagate (-1:all, >=1:nb)', default=-1},
      {arg='sizeAverage', type='number', help='if true, forward() returns an average instead of a sum of errors', default=true}
   )
end

function SpatialClassNLLCriterion:adjustTarget(input, target)
   -- (1) if the target map has an incorrect size, it is assumed
   --     to be at the original scale of the data (e.g. for dense
   --     classification problems, like scene parsing, the target
   --     map is at the resolution of the input image. Now the input
   --     of this criterion is the output of some neural network, 
   --     and might have a smaller size/resolution than the original
   --     input). Step (2) corrects for convolutional-induced losses,
   --     while step (3) corrects for downsampling/strides.
   local sratio = self.resampleTarget
   if (target:size(1)*sratio) ~= input:size(2) then
      local h = input:size(1)/sratio
      local y = math.floor((target:size(1) - (input:size(1)-1)*1/sratio)/2) + 1
      local w = input:size(2)/sratio
      local x = math.floor((target:size(2) - (input:size(2)-1)*1/sratio)/2) + 1
      target = target:narrow(1,y,h):narrow(2,x,w)
   end
   -- (2) correct target by resampling it to the size of the 
   --     input. this is to compensate for downsampling/pooling
   --     operations.
   if sratio ~= 1 then
      local target_scaled = torch.Tensor(input:size(2), input:size(3))
      image.scale(target, target_scaled, 'simple')
      target = target_scaled
   end
   self.target = target
   return target
end

function SpatialClassNLLCriterion:updateOutput(input,target)
   -- (1) adjust target: class -> distributions of classes
   --                    compensate for convolution losses
   --                    compensate for striding effects
   --                    ignore a classe
   target = self:adjustTarget(input, target)
   -- (2) the full output contains as many errors as input
   --     vectors, whereas the self.output is a scalar that
   --     prunes all the errors
   self.fullOutput = self.fullOutput or torch.Tensor()
   self.fullOutput:resizeAs(target)
   -- (3) compute the dense errors:
   input.nn.SpatialClassNLLCriterion_updateOutput(self,input,target)
   -- (4) prune the errors, either by averaging, or accumulation:
   if self.sizeAverage then
      self.output = self.fullOutput:mean()
   else
      self.output = self.fullOutput:sum()
   end
   return self.output
end

function SpatialClassNLLCriterion:updateGradInput(input,target)
   -- (1) retrieve adjusted target
   target = self.target
   -- (2) resize input gradient map
   self.gradInput:resizeAs(input):zero()
   -- (3) compute input gradients, based on the nbGradients param
   if self.nbGradients == -1 then
      -- dense gradients
      input.nn.SpatialClassNLLCriterion_updateGradInput(self,input,target,self.gradInput)
   elseif self.nbGradients == 1 then
      -- only 1 gradient is computed, sampled in the center
      self.fullGradInput = torch.Tensor() or self.fullGradInput
      self.fullGradInput:resizeAs(input):zero()
      input.nn.SpatialClassNLLCriterion_updateGradInput(self,input,target,self.fullGradInput)
      local y = math.ceil(self.gradInput:size(2)/2)
      local x = math.ceil(self.gradInput:size(3)/2)
      self.gradInput:select(3,x):select(2,y):copy(self.fullGradInput:select(3,x):select(2,y))
   else
      -- only N gradients are computed, sampled in random locations
      self.fullGradInput = torch.Tensor() or self.fullGradInput
      self.fullGradInput:resizeAs(input):zero()
      input.nn.SpatialClassNLLCriterion_updateGradInput(self,input,target,self.fullGradInput)
      for i = 1,self.nbGradients do
         local x = math.random(1,self.gradInput:size(1))
         local y = math.random(1,self.gradInput:size(2))
         self.gradInput:select(3,x):select(2,y):copy(self.fullGradInput:select(3,x):select(2,y))
      end
   end
   return self.gradInput
end
