local SGD,parent = torch.class('nn.SGDOptimization', 'nn.BatchOptimization')

function SGD:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
      'SGDOptimization', nil,
      {arg='maxIterations', type='number', 
       help='maximum nb of iterations per pass', default=1},
      {arg='learningRate', type='number', 
       help='learning rate (W = W - rate*dE/dW)', default=1e-2},
      {arg='learningRateDecay', type='number', 
       help='learning rate decay (lr_t = lr_0 / (1 + samplesSeen*lrDecay))', default=0},
      {arg='weightDecay', type='number', 
       help='amount of weight decay (W = W - decay*W)', default=0},
      {arg='momentum', type='number', 
       help='amount of momentum on weights (dE/W = dE/dW*(1-momentum) + prev(dE/dW)*momentum)', default=0}
   )
end

function SGD:optimize()
   -- optimize N times
   for i = 1,self.maxIterations do
      -- (0) evaluate f(X) + df/dX
      self.evaluate()

      -- (1) apply momentum
      if self.momentum ~= 0 then
         if not self.currentGradParameters then
            self.currentGradParameters = torch.Tensor():resizeAs(self.gradParameters):copy(self.gradParameters)
         else
            self.currentGradParameters:mul(self.momentum):add(1-self.momentum, self.gradParameters)
         end
      else
         self.currentGradParameters = self.gradParameters
      end

      -- (2) weight decay
      if self.weightDecay ~= 0 then
         self.parameters:add(-self.weightDecay, self.parameters)
      end

      -- (3) learning rate decay (annealing)
      local learningRate = 
         self.learningRate / (1 + self.sampleCounter*self.learningRateDecay)
      
      -- (4) parameter update with single or individual learningRates
      if self.learningRates then
         -- we are using diagHessian and have individual learningRates
         self.deltaParameters = self.deltaParameters or 
            torch.Tensor():typeAs(self.parameters):resizeAs(self.currentGradParameters)
         self.deltaParameters:copy(self.learningRates):cmul(self.currentGradParameters)
         self.parameters:add(-learningRate, self.deltaParameters)
      else
         -- normal single learningRate parameter update
         self.parameters:add(-learningRate, self.currentGradParameters)
      end

      -- (5) allreduce sync
      if self.allreduce then
         if (self.sampleCounter % self.allreduceSyncTime) == self.allreduceSyncTime-1 then
            allreduce.best(self.parameters, self.accError)
            self.accError = 0
         else
            self.accError = self.accError + self.output 
         end
      end
   end -- for loop on maxIterations
end

function SGD:condition (inputs, targets, ctype)
   if (ctype == 'dh') then
      -- Leon and Antoines' SGD-QN algorithm
      self:diagHessian(inputs,targets)
   elseif (ctype == 'qn') then
      -- Leon and Antoines' SGD-QN algorithm
      self:QN(inputs,targets)
   elseif (ctype == 'olr') then 
      -- Yann's optimal learning rate from Efficient BackProp 1998
      self:optimalLearningRate(inputs, targets)
   else 
      print("Not contitioning : don't understand conditioning type")
   end
end

function SGD:QN(inputs, targets)
   
end

function SGD:diagHessian(inputs, targets)
   if not self.learningRates then 
      print('<SGD> creating learningRates, initDiagHessian')
      -- do initialization
      self.diagHessianEpsilon = self.diagHessianEpsilon or 1e-2
      self.learningRates = torch.Tensor():typeAs(self.parameters):resizeAs(self.parameters):fill(1)
      -- we can call this multiple times as it will only create the tensors once.
      self.module:initDiagHessianParameters()
      self.diagHessianParameters = 
         nnx.flattenParameters(nnx.getDiagHessianParameters(self.module))
   end
   -- reset gradients
   self.gradParameters:zero()
   -- reset Hessian Parameterns
   self.diagHessianParameters:zero()
   -- reset individual learningRates
   self.learningRates:fill(1)
   -- estimate diag hessian over dataset
   if type(inputs) == 'table' then      -- slow
      for i = 1,#inputs do
         local output = self.module:forward(inputs[i])
         local critDiagHessian = 
            self.criterion:updateDiagHessianInput(output, targets[i])
         self.module:updateDiagHessianInput(inputs[i], critDiagHessian)
         self.module:accDiagHessianParameters(inputs[i], critDiagHessian)
      end
      self.diagHessianParameters:div(#inputs)
   else
      local output = self.module:forward(inputs)
      -- not sure if we can do the fast version yet
      local critDiagHessian = criterion:updateDiagHessianInput(output, targets)
      module:updateDiagHessianInput(inputs, critDiagHessian)
      module:accDiagHessianParameters(inputs, critDiagHessian)
      self.diagHessianParameters:div(inputs:size(1))
   end
   print('<diagHessian>')
   print(' + before max ')
   print(' + epsilon: '..self.diagHessianEpsilon)
   print(' + norm of dhP: '..self.diagHessianParameters:norm())
   print(' + max dhP : '..self.diagHessianParameters:max())
   print(' + min dhp: '.. self.diagHessianParameters:min())
  -- protect diag hessian 
   self.diagHessianParameters:apply(
      function(x)
	 local out = math.max(math.abs(x), self.diagHessianEpsilon) 
	 if (x < 0) then out = -out end
	 return out
      end)

   -- now learning rates are obtained like this:
   self.learningRates:cdiv(self.diagHessianParameters) 
   -- test 
   print(' + after max')
   print(' + norm of dhP: '..self.diagHessianParameters:norm()..
      ' norm of LR: '..self.learningRates:norm())
   print(' + max dhP : '..self.diagHessianParameters:max() .. 
      ' min LR: '..self.learningRates:min())
   print(' + min dhp: '.. self.diagHessianParameters:min() ..
      ' max LR: '..self.learningRates:max())
   -- self.learningRates:div(self.learningRates:norm())
end

function SGD:optimalLearningRate(inputs, targets)
   
   -- conditioning using Yann's optimal learning rate
   -- from Efficient BackProp 1998
   -- self.alpha = self.alpha or 1e-2 -- 1 / ||parameters|| ?
   self.alpha = self.alpha or 1e-2 -- 1 / ||parameters|| ?
   self.gamma = self.gamma or 0.95
      
   if not self.phi then
      -- make tensor in current default type
      self.phi = torch.Tensor(self.gradParameters:size())
      -- no lab functions for CudaTensors so
      local old_type = torch.getdefaulttensortype()
      if (old_type == 'torch.CudaTensor') then
	 torch.setdefaulttensortype('torch.FloatTensor')
      end
      local r = torch.randn(self.gradParameters:size())
      r:div(r:norm()) -- norm 2
      if (old_type == 'torch.CudaTensor') then
	 torch.setdefaulttensortype(old_type)
      end
      self.phi:copy(r)
   end

   -- scratch vectors which we don't want to re-allocate every time
   self.param_bkup = self.param_bkup or torch.Tensor():resizeAs(self.parameters)
   self.grad_bkup = self.grad_bkup or torch.Tensor():resizeAs(self.gradParameters)
   -- single batch (not running average version)

   if type(inputs) == 'table' then      -- slow
      print("<SGD conditioning> slow version ")
      -- (1) compute dE/dw(w)
      -- reset gradients
      self.gradParameters:zero()
      for i = 1,#inputs do
	 -- estimate f
	 local output = self.module:forward(inputs[i])
	 local err  = self.criterion:forward(output, targets[i])
	 -- estimate df/dW
	 local df_do = self.criterion:backward(output, targets[i])
	 self.module:backward(inputs[i], df_do)
      end
      -- normalize gradients
      -- self.gradParameters:div(#inputs)
      
      -- backup gradient and weights
      self.param_bkup:copy(self.parameters)
      self.grad_bkup:copy(self.gradParameters)
      
      -- (2) compute dE/dw(w + alpha * phi / || phi|| )
      -- normalize + scale phi
      local norm_phi = self.phi:norm()
      print(' + norm phi before: ',norm_phi,' alpha: ',self.alpha)
      if norm_phi > 1e-16 then
         self.phi:div(self.phi:norm()):mul(self.alpha)
      else
         self.phi:fill(1/self.phi:size(1)):mul(self.alpha)
      end
      norm_phi = self.phi:norm()
      print(' + norm phi after : ', norm_phi)
      -- perturb weights
      print(' + norm param before wiggle: ',self.parameters:norm())
      self.parameters:add(self.phi)
      print(' + norm param after  wiggle: ',self.parameters:norm())
      -- reset gradients
      self.gradParameters:zero()
      --re-estimate f
      for i = 1,#inputs do
	 -- estimate f
	 output = self.module:forward(inputs[i])
	 err  = self.criterion:forward(output, targets[i])
	 -- estimate df/dW
	 df_do = self.criterion:backward(output, targets[i])
	 self.module:backward(inputs[i], df_do)
      end
      -- normalize gradients
      -- self.gradParameters:div(#inputs)

      -- (3) phi - 1/alpha(dE/dw(w + alpha * oldphi / || oldphi ||) - dE/dw(w))
      -- compute new phi
      self.phi:copy(self.grad_bkup):mul(-1):add(self.gradParameters):mul(1/self.alpha)
      norm_phi = self.phi:norm()
      print(' + norm old_grad: ',self.grad_bkup:norm())
      print(' + norm cur_grad: ',self.gradParameters:norm())
      print(' + norm phi: ',norm_phi)      
      -- (4) new learning rate eta = 1 / || phi ||
      if norm_phi > 0 then
         self.learningRate = 1 / ( norm_phi * #inputs )
      else
         self.learningRate = 1e-4
      end
      print(' + conditioned learningRate: ', self.learningRate)
      -- (5) reset parameters and zero gradients
      self.parameters:copy(self.param_bkup)
      self.gradParameters:zero()
   else -- fast
      -- (1) compute dE/dw(w)
      -- reset gradients
      self.gradParameters:zero()
      -- estimate f
      local output = self.module:forward(inputs)
      local err  = self.criterion:forward(output, targets)
      -- estimate df/dW
      local df_do = self.criterion:backward(output, targets)
      self.module:backward(inputs, df_do)
      -- backup gradient and weights
      self.param_bkup:copy(self.parameters)
      self.grad_bkup:copy(self.gradParameters)
      -- divide by number of samples
      -- self.grad_bkup:div(inputs:size(1))

      -- (2) compute dE/dw(w + alpha * phi / || phi|| )
      -- normalize + scale phi
      print('norm phi before: ',self.phi:norm(),' alpha: ',self.alpha)
      self.phi:div(self.phi:norm()):mul(self.alpha)
      print('norm phi after: ',self.phi:norm())
      -- perturb weights
      print('norm param before: ',self.parameters:norm())
      self.parameters:add(self.phi)
      print('norm param after: ',self.parameters:norm())
      -- reset gradients
      self.gradParameters:zero()
      --re-estimate f
      output = self.module:forward(inputs)
      self.output = self.criterion:forward(output, targets)
      -- re-estimate df/dW
      df_do = self.criterion:backward(output, targets)
      self.module:backward(inputs, df_do)
      -- self.gradParameters:div(inputs:size(1))

      -- (3) phi - 1/alpha(dE/dw(w + alpha * oldphi / || oldphi ||) - dE/dw(w))
      -- compute new phi
      if true then
	 -- running average
	 self.phi:mul(self.gamma):add(self.grad_bkup):mul(-1):add(self.gradParameters):mul(1/self.alpha)
      else 
	 self.phi:copy(self.grad_bkup):mul(-1):add(self.gradParameters):mul(1/self.alpha)
      end
      print('norm old_grad: ',self.grad_bkup:norm(),' norm cur_grad: ',self.gradParameters:norm(), ' norm phi: ',self.phi:norm())      
      -- (4) new learning rate eta = 1 / || phi || 
      self.learningRate = 1 / self.phi:norm()
      -- (5) reset parameters and zero gradients
      self.parameters:copy(self.param_bkup)
      self.gradParameters:zero()
   end 
end