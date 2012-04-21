local SNES,parent = torch.class('nn.SNESOptimization', 'nn.BatchOptimization')

function SNES:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
                     'SNESOptimization', nil,
                     {arg='lambda', type='number', help='number of parallel samples', default=100},
                     {arg='mu_0', type='number', help='initial value for mu', default=0},
                     {arg='sigma_0', type='number', help='initial value for sigma', default=1},
                     {arg='eta_mu', type='number', help='learning rate for mu', default=1},
                     {arg='eta_sigma', type='number', help='learning rate for sigma', default=1e-3}
                  )
   -- original parameters
   self.parameter = nnx.flattenParameters(nnx.getParameters(self.module))
   -- SNES needs one module per lambda
   self.modules = {}
   self.criterions = {}
   self.parameters = {}
   for i = 1,self.lambda do
      self.modules[i] = self.module:clone()
      self.criterions[i] = self.criterion:clone()
      self.parameters[i] = nnx.flattenParameters(nnx.getParameters(self.modules[i]))
   end
   -- SNES initial parameters
   self.mu = torch.Tensor(#self.parameters[1]):fill(self.mu_0)
   self.sigma = torch.Tensor(#self.parameters[1]):fill(self.sigma_0)
   -- SNES gradient vectors
   self.gradmu = torch.Tensor():resizeAs(self.mu)
   self.gradsigma = torch.Tensor():resizeAs(self.sigma)
   -- SNES utilities
   self:utilities()
end

function SNES:f(th, X, inputs, targets)
   -- set parameter to X
   self.parameters[th]:copy(X)
   -- estimate f on given mini batch
   local f = 0
   for i = 1,#inputs do
      local output = self.modules[th]:forward(inputs[i])
      f = f + self.criterions[th]:forward(output, targets[i])
   end
   f = f/#inputs
   return f
end

function SNES:utilities()
   -- compute utilities
   local sum = 0
   self.u = {}
   for i = 1,self.lambda do
      local x = i/self.lambda -- x in [0..1]
      self.u[i] = math.exp((1-x)*10)-1
      sum = sum + self.u[i]
   end
   -- normalize u
   for i = 1,self.lambda do
      self.u[i] = self.u[i] / sum
   end
end

function SNES:optimize(inputs, targets)
   -- fitness for each sample drawn
   local fitness = {}

   -- draw samples
   for i = 1,self.lambda do
      -- random distribution
      local s_k = torch.randn(self.sigma:size())
      local z_k = self.sigma:clone():cmul(s_k):add(self.mu)

      -- evaluate fitness of f(X)
      local f_X = self:f(i, z_k, inputs, targets)

      -- store s_k, z_k
      fitness[i] = {f=f_X, s=s_k, z=z_k}
   end

   -- sort fitness tables
   table.sort(fitness, function(a,b) if a.f < b.f then return a end end)

   -- set current output to best f_X (lowest)
   self.output = fitness[1].f

   -- compute gradients
   self.gradmu:zero()
   self.gradsigma:zero()
   for i = 1,self.lambda do
      self.gradmu:add(self.u[i], fitness[i].s)
      self.gradsigma:add(self.u[i], torch.pow(fitness[i].s,2):add(-1))
   end

   -- update parameters
   for i = 1,self.lambda do
      self.mu:add(self.eta_mu, self.sigma:clone():cmul(self.gradmu))
      self.sigma:cmul(torch.exp(self.gradsigma:clone():mul(self.eta_sigma/2)))
   end

   -- optimization done, copy back best parameter vector
   self.parameter:copy(self.mu)

   -- verbose
   self.batchCounter = self.batchCounter or 0
   self.batchCounter = self.batchCounter + 1
   if self.verbose >= 2 then
      print('<SNESOptimization> evaluated f(X) on ' .. self.lambda .. ' random points')
      print('  + batches seen: ' .. self.batchCounter)
      print('  + lowest eval f(X) = ' .. fitness[1].f)
      print('  + worst eval f(X) = ' .. fitness[#fitness].f)
   end

   -- for now call GC
   collectgarbage()
end
