local ASGD,parent = torch.class('nn.ASGDOptimization', 'nn.SGDOptimization')

-- ASGD: 
--     w := (1 - lambda eta_t) w - eta_t dL/dw(z,w)
--     a := a + mu_t [ w - a ]
--
--  eta_t = eta_0 / (1 + lambda eta0 t) ^ 0.75
--   mu_t = 1/max(1,t-t0)
-- 
-- implements ASGD algoritm as in L.Bottou's sgd-2.0

function ASGD:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
     'ASGDOptimization', nil,
     {arg='eta0', type='number',
      help='eta0 parameter for ASGD', default=1e-4},
     {arg='t0', type='number',
      help='point at which to start averaging', default=1e6},
     {arg='lambda', type='number',
      help='lambda for ASGD --decay term', default=1},
     {arg='alpha', type='number',
      help='alpha for ASGD -- power for eta update', default=0.75}
  )
   self.eta_t = self.eta0
   self.mu_t  = 1
   self.t     = 0
end

function ASGD:optimize() 
   -- (0) evaluate f(X) + df/dX
   self.evaluate()
   -- (1) decay term  
   --     w := (1 - lambda eta_t) w
   self.parameters:mul(1 - self.lambda * self.eta_t)
   -- (2) parameter update with single or individual learningRates
   --     w += - eta_t dL/dw(z,w)
   if self.learningRates then
      -- we are using diagHessian and have individual learningRates
      self.deltaParameters = self.deltaParameters or 
         self.parameters.new():resizeAs(self.gradParameters)
      self.deltaParameters:copy(self.learningRates):cmul(self.gradParameters)
      self.parameters:add(-self.eta_t, self.deltaParameters)
   else
      -- normal single learningRate parameter update
      self.parameters:add(-self.eta_t, self.gradParameters)
   end
   -- (3) Average part
   --     a := a + mu_t [ w - a ]
   self.a = self.a or self.parameters.new():resizeAs(self.parameters):zero()
   if self.mu_t ~= 1 then
      self.tmp = self.tmp or self.a.new():resizeAs(self.a)
      self.tmp:copy(self.parameters):add(-1,self.a):mul(self.mu_t)
      self.a:add(self.tmp)
   else 
      self.a:copy(self.parameters)
   end
   -- (4) update eta_t and mu_t
   -- (4a) increment time counter
   self.t = self.t + 1
   -- (4b) update eta_t
   --  eta_t = eta_0 / (1 + lambda eta0 t) ^ 0.75
   self.eta_t = self.eta0 / math.pow((1 + self.lambda * self.eta0 * self.t ),0.75)
   -- (4c) update mu_t
   --   mu_t = 1/max(1,t-t0)
   self.mu_t = 1 / math.max(1,self.t - self.t0)
end

-- in ASGD we keep a copy of the parameters which is an averaged
-- version of the current parameters.  This function is to test with
-- those averaged parameters.  Best to run on batches because we have
-- to copy the full parameter vector

function ASGD:test(_inputs, _targets) -- function test
   -- (0) make a backup of the online parameters
   self.backup = self.backup or
     self.parameters.new():resizeAs(self.parameters)
   self.backup:copy(self.parameters)
   -- (1) copy average parameters into the model
   self.parameters:copy(self.a)
   -- (2) do the test with the average parameters
   self.output = self.module:forward(_inputs)
   self.error  = self.criterion:forward(self.output, _targets)
   -- (3) copy back the online parameters to continue training
   self.parameters:copy(self.backup)
   return self.error
end
