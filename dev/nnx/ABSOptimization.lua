local ABS,parent = torch.class('nn.ABSOptimization', 'nn.SGDOptimization')

-- ABS is Adaptive Batch Size

function ABS:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
                     'ABSOptimization', nil,
                     {arg='theta', type='number',
                      help='threshold for increasing batch size', default=1}
                  )
end

function ABS:reduce_hook()
   -- standard reduce is to sum the gradients
   -- accumulate partial gradients, and average
   self.gradParameters:zero()
   -- compute mean from the batches
   for t = 1,P do
      self.gradParameters:add(gradParametersPartial[t])
   end
   self.gradParameters:div(#inputs)
   self.gradStd = torch.Tensor():resizeAs(self.gradParameters):zero()
   local tmp = torch.Tensor():resizeAs(self.gradParameters):zero()
   -- compute std
   -- 1) sum the variances (doing it element-wise)
   for t = 1,P do
      tmp:add(self.gradParameters):mul(-1):add(gradParametersPartial[t]):pow(2)
      self.gradStd:add(tmp)
      tmp:zero()
   end
   -- 2) now take sqrt
   self.gradStd:sqrt()

   -- test to increase batchSize
   if self.theta * self.gradStd:norm() > self.gradParameters:norm() then
      self.batchSize = self.batchSize * 2
   end
   -- return average f(X)
   self.output = 0
   for t = 1,P do
      self.output = self.output + outputsPartial[t]
   end
   self.output = self.output/#inputs
end
