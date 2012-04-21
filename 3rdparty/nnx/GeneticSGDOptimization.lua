local GenSGD,parent = torch.class('nn.GeneticSGDOptimization',
                                  'nn.BatchOptimization')

-- this module parallelizes SGD in a particular way.  It sends out the
-- same batch to each of several workers, each with a different learning
-- rate.  The workers run and the parameters from the best worker and
-- it's learning rate are kept for the next batch.

function GenSGD:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
                     'GenSGDOptimization', nil,
                     {arg='maxIterations', type='number',
                      help='maximum nb of iterations per pass', default=1},
                     {arg='learningRate', type='number',
                      help='learning rate (W = W - rate*dE/dW)', default=1e-2},
                     {arg='learningRateDecay', type='number',
                      help='learning rate decay (lr_t = lr_0 / (1 + samplesSeen*lrDecay))',
                      default=0},
                     {arg='weightDecay', type='number',
                      help='amount of weight decay (W = W - decay*W)', default=0},
                     {arg='momentum', type='number',
                      help='amount of momentum on weights (dE/W = dE/dW*(1-momentum) + prev(dE/dW)*momentum)', default=0},
                     {arg='sigma', type='number',
                      help='octaves over which to search when randomizing learning rate',default=100},
                     {arg='gamma', type='number',
                      help='mixing factor with old learning rate', default=0.8},
                     {arg='dist',type='string',
                      help='type of distribution "loguniform" (default) or "lognormal"',
                      default='loguniform'},
                     {arg='adaptive_batchSize', type='boolean',
                      help='do Chin-Nocedal test of gradient for increasing the batch size',
                      default=false},
                     {arg='theta', type='number',
                      help='threshold for increasing batch size', default=1},
                     {arg='exact_output', type='boolean',
                      help='recompute output on batch using final parameters',
                      default=false},
                     {arg='exact_batchSize', type='boolean',
                      help='recompute gradParameters on batch using final parameters',
                      default=false}
                  )
   if self.parallelize < 2 then
      xerror('GenSGD needs to work on several processors: set parallelize',
             'nn.GenSGDOptimization')
   end
   -- change the mapper to send the same batch to each worker
   self.copyBatch = true
   -- create default parameter set which will be randomized for each worker
   self.baseParameters = { momentum           = self.momentum,
                           weightDecay        = self.weightDecay,
                           learningRate       = self.learningRate,
                           learningRateDecay  = self.learningRateDecay,
                           sampleCounter      = self.sampleCounter,
                           adaptive_batchSize = self.adaptive_batchSize,
                           theta              = self.theta,
                           exact_output       = self.exact_output,
                           exact_batchSize    = self.exact_batchsize
                        }
end

-- log normal
--
-- + mean is transformed to be the peak of the distribution u.
-- + sigma is not the stdev.
--
function lognormal(n,mean,sigma)
   -- pdf = lambda s,m,x: exp(-(log(x)-m)**2 / (2.*s**2)) / ( x*sqrt(2.*pi*s**2) )
   local u = -1 * (math.log(mean) - (sigma * sigma * 0.5))
   local x = torch.rand(n)
   local y = torch.Tensor():resizeAs(x):copy(x)
   y:log():add(u)
   y:cmul(y):mul(-1):div(2*sigma*sigma):exp()
   x:mul(sigma*math.sqrt(2*math.pi))
   return y:cdiv(x)
end

-- log uniform
-- returns n values uniformly distributed in log space between min and max
function loguniform (n,rate,octaves)
   local a = math.log(rate/octaves)
   local b = math.log(rate*octaves)-a
   return torch.rand(n):mul(b):add(a):exp()
end

-- we are changing the way we map and reduce.  It would be nice to
-- change gradParametersPartial to ParametersPartial, as the logic is
-- different for this kind of parallelization.
function GenSGD:map_hook()
   local P = self.parallelize
   -- transmit new parameters to all workers
   self.children:join()
   self.children:send(self.parameters)
   -- randomize learning rate (could randomize other bits).  Using a
   -- log normal around the base rate.
   local n  = torch:Tensor()
   if self.dist == 'lognormal' then
      n = lognormal(P, self.learningRate, self.sigma)
   else
      n = loguniform(P, self.learningRate,self.sigma)
   end

   self.baseParameters.sampleCounter = self.sampleCounter
   for t = 1,P do
      self.baseParameters.learningRate = n[t]
      self.children[t]:send(self.baseParameters)
   end
   -- then wait for all workers to return their Parameters + outputs
   -- should rename this to parametersParallel and optionsParallel
   gradParametersPartial = self.children:receive()
   outputsPartial = self.children:receive()
   print('rates , results')
   for t = 1,P do
      print(n[t],outputsPartial[t].f_x)
   end
   -- force cleanup
   collectgarbage()
end

function GenSGD:reduce_hook()
   local P = self.parallelize
   local id = 0
   local mx = 1e9
   for t = 1,P do
      if outputsPartial[t].f_x < mx then
         id = t
         mx = outputsPartial[t].f_x
      end
   end
   if id == 0 then
      xerror('diverging','nn.GenSGDOptimization')
   else
      self.baseParameters = outputsPartial[id]
      self.learningRate =
         self.gamma*self.learningRate +
         (1-self.gamma)*self.baseParameters.learningRate
      self.output = self.baseParameters.f_x
      print('chose: '..self.baseParameters.learningRate..' b/c '..self.output)
      print('new LR: '..self.learningRate)
      -- in this case we get the parameters back directly
      self.parameters:copy(gradParametersPartial[id])
   end
end

function GenSGD:optimize()
   self.evaluate()
end

-- optimization (could do others in this mode)
GenSGD.optimizer =
   function (module,params)
      -- apply momentum (store in the module)
      if params.momentum ~= 0 then
         if not module.currentGradParameters then
            module.currentGradParameters =
               torch.Tensor():resizeAs(module.gradParameters):copy(module.gradParameters)
         else
            module.currentGradParameters:mul(params.momentum):add(1-params.momentum, module.gradParameters)
         end
      else
         module.currentGradParameters = module.gradParameters
      end

      -- weight decay
      if params.weightDecay ~= 0 then
         module.parameters:add(-params.weightDecay, module.parameters)
      end

      -- update parameters
      local learningRate =
         params.learningRate / (1 + params.sampleCounter*params.learningRateDecay)
      module.parameters:add(-learningRate, module.currentGradParameters)
      -- make keep track of final rate
      params.learningRate = learningRate
   end

function GenSGD:setup_mapreduce ()
   -- (0) startup parallel package
   if not xrequire 'parallel' then
      xerror('install parallel for Lua to enable parallel computing (luarocks install parallel)',
             'nn.GenSGDOptimization')
   end
   local worker_code  =
      function()
         -- require packages
         require 'nnx'

         -- retrieve optional code to setup worker
         precode = parallel.parent:receive()
         if type(precode) == 'function' then precode() end

         -- retrieve module + criterion + optimimzer at startup
         parallel.yield()

         module    = parallel.parent:receive()
         criterion = parallel.parent:receive()
         optimizer = parallel.parent:receive()

         -- retrieve optional prehook/posthook
         prehook = parallel.parent:receive()
         posthook = parallel.parent:receive()
         if type(prehook) ~= 'function' then prehook = nil end
         if type(posthook) ~= 'function' then posthook = nil end

         -- I don't understand this [MS]
         -- get pointer to parameter and gradParameter vectors
         -- (this assumes that parameters+gradParameters are already flat parameters:
         --  it should be the case, as the parent process flattens them at __init)
         function check(tocheck)
            for i = 2,#tocheck do
               if tocheck[i]:storage() ~= tocheck[i-1]:storage() then
                  print('<BatchOptimization> error: inconsistent parameter vector (not flat)')
                  return
               end
            end
         end
         tableParameters = nnx.getParameters(module)
         tableGradParameters = nnx.getGradParameters(module)
         check(tableParameters)
         check(tableGradParameters)
         parameters = torch.Tensor():set(tableParameters[1]:storage())
         gradParameters = torch.Tensor():set(tableGradParameters[1]:storage())

         -- outer loop: mini-batches
         while true do
            -- sync
            if parallel.yield() == 'break' then break end

            -- receive new mini-batch
            inputs  = parallel.parent:receive()
            targets = parallel.parent:receive()
            options = parallel.parent:receive()

            -- inner loop: evaluations
            while true do
               -- sync
               if parallel.yield() == 'break' then break end

               -- receive new set of parameters
               parameters:copy(parallel.parent:receive())
               -- receive the learning rate etc. parameters which are
               -- tweaked for each thread
               opt_param = parallel.parent:receive()

               -- evaluate gradients on inputs for this thread and perform
               -- SGD on these inputs

               module.parameters = parameters
               module.gradParameters = gradParameters

               -- used for the adaptive batch sizes
               local partialGrads = torch.Tensor()
               if opt_param.adaptive_batchSize then
                  -- this could be huge so be careful
                  partialGrads:resize(#inputs,gradParameters:size(1))
               end
               local f_x = 0
               -- FIXME implement maxIterations here
               for i = 1,#inputs do
                  -- reset gradients
                  gradParameters:zero()
                  -- estimate f
                  local output = module:forward(inputs[i])
                  local err = criterion:forward(output, targets[i])
                  f_x = f_x + err
                  -- estimate df/dW
                  local df_do = criterion:backward(output, targets[i])
                  module:backward(inputs[i], df_do)
                  optimizer(module,opt_param)
                  if opt_param.adaptive_batchSize and
                     not opt_param.exact_batchSize then
                     partialGrads[i]:copy(gradParameters)
                  end
               end
               -- if we need the result averaged over all the samples _after_
               -- the gradient steps we must do one more loop to fprop through
               -- the samples and collect the error _after_ the optimization
               if  opt_param.exact_output then
                  f_x = 0 -- reset
                  for i = 1,#inputs do
                     gradParameters:zero()
                     -- estimate f
                     local output = module:forward(inputs[i])
                     local err = criterion:forward(output, targets[i])
                     f_x = f_x + err
                     -- if adjust batch size (recompute individual
                     -- gradients) for the final parameters
                     if opt_param.adaptive_batchSize and opt_param.exact_batchsize then
                        local df_do = criterion:backward(output, targets[i])
                        module:backward(inputs[i], df_do)
                        partialGrads[i]:copy(gradParameters)
                     end
                  end
               end
               -- in this case send back parameters themselves b/c they are
               -- already optimized
               parallel.parent:send(parameters)
               -- need to make sure we keep track of what was used to
               -- compute these params along with the outputs
               opt_param['f_x'] = f_x/#inputs
               if opt_param.adaptive_batchsize then
                  local gradStd = torch.Tensor():resizeAs(gradParameters):zero()
                  -- take componentwise std
                  for i = 1,gradStd:size(1) do
                     gradStd[i] = partialGrads:narrow(2,i,1):std()
                  end
                  -- test to increase batchSize
                  if opt_param.theta * gradStd:norm() > gradParameters:norm() then
                     opt_param['batchSize'] = #inputs*2
                  else
                     opt_param['batchSize'] = #inputs
                  end
               end
               parallel.parent:send(opt_param)
               -- force cleanup
               collectgarbage()
            end
         end
      end

   local setup = function()
                    -- (1) optional calibration
                    if parallel.remotes then
                       parallel.calibrate()
                    end

                    -- (2) startup all workers
                    self.children = parallel.sfork(self.parallelize)
                    self.children:exec(worker_code)

                    -- (3) send them optional config code
                    self.children:send(self.precode or '')

                    -- (4) and send them the module + criterion architecture
                    self.children:join()
                    self.children:send(self.module)
                    self.children:send(self.criterion)
                    self.children:send(self.optimizer)
                 end

   local ok,err = pcall(setup)
   if not ok then parallel.close() error(err) end
end
