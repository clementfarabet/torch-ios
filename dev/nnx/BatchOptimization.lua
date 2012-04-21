local Batch,parent = torch.class('nn.BatchOptimization', 'nn.Optimization')

-- this is a generic class for any batch optimization modeled after
-- the LBFGS optimization.  It simply provides a batch.evaluate() method
-- which creates a self.parameters and self.gradParameters from your
-- self.module

function Batch:__init(...)
   parent.__init(self)
   xlua.unpack_class(self, {...},
                     'BatchOptimization', nil,
                     {arg='module', type='nn.Module', help='a module to train', req=true},
                     {arg='criterion', type='nn.Criterion',
                      help='a criterion to estimate the error', req=true},
                     {arg='parallelize', type='number',
                      help='parallelize onto N cores (experimental!)', default=1},
                     {arg='precode', type='function',
                      help='optional code to be run by each parallel worker at their init'},
                     {arg='verbose', type='number',
                      help='verbose level during training [0-2]', default=0},
                     {arg='allreduce', type='boolean', help='use allreduce', default=false},
                     {arg='allreduceSyncTime', type='boolean', help='sync period', default=1},
                     {arg='allreduceMaster', type='string', help='master address', default='localhost'},
                     {arg='allreduceUniqueId', type='boolean', help='job unique id', default=0},
                     {arg='allreduceNbNodes', type='boolean', help='number of nodes', default=1},
                     {arg='allreduceNodeId', type='boolean', help='this node\'s id', default=1}
                  )
   self.parameters = nnx.flattenParameters(nnx.getParameters(self.module))
   self.gradParameters = nnx.flattenParameters(nnx.getGradParameters(self.module))

   self.evalCounter   = 0
   self.batchCounter  = 0
   self.sampleCounter = 0

   if self.parallelize > 1 then
      self:setup_mapreduce()
   end
   self.P = self.parallelize

   if self.allreduce then
      xrequire 'allreduce'
      allreduce.init(self.allreduceMaster, self.allreduceUniqueId, 
                     self.allreduceNbNodes, self.allreduceNodeId)
      self.accError = 0
   end
end

function Batch:forward(inputs, targets, options)
   options = options or {}
   targets = targets or {}
   if self.P > 1 then
      return self:forward_mapreduce(inputs, targets, options)
   else
      return self:forward_sequential(inputs, targets, options)
   end
end

function Batch:forward_sequential(inputs, targets, options)
   -- (0) batch size
   local batchsize = 1
   if type(inputs) == 'table' then
      batchsize = #inputs
   else
      batchsize = inputs:size(1)
   end

   -- (1) construct a closure that compute f(inputs) + df/dW
   --     after each call to that function:
   --       + self.parameters contains the current X vector
   --       + self.gradParameters contains the estimated dF/dX vector
   --       + self.output contains the estimated (average) F(X)
   self.evaluate
      = function()
           -- verbose
           if self.verbose >= 2 then
              print('<BatchOptimization> evaluating f(X) + df/dX')
           end
           local _t_ = sys.clock()

           -- reset gradients
           self.gradParameters:zero()

           -- f is the average of all criterions
           self.output = 0

           -- minibatch
           if type(inputs) == 'table' then
              -- given all inputs, evaluate gradients
              for i = 1,#inputs do
                 -- user hook
                 if self.prehook then
                    self.prehook(self, {inputs[i], targets[i], options[i]})
                 end
                 -- estimate f
                 local output = self.module:forward(inputs[i])
                 local err = self.criterion:forward(output, targets[i])
                 self.output = self.output + err
                 -- estimate df/dW
                 local df_do = self.criterion:backward(output, targets[i])
                 self.module:backward(inputs[i], df_do)
                 -- user hook
                 if self.posthook then
                    self.posthook(self, {inputs[i], targets[i], options[i]})
                 end
                 -- update evaluation counter
                 self.evalCounter = self.evalCounter + 1
              end

              -- normalize gradients and f(X)
              self.gradParameters:div(batchsize)
              self.output = self.output/batchsize

           else -- minibatch is assumed to be a BatchSize x ... tensor
              -- estimate f
              local output = self.module:forward(inputs)
              self.output = self.criterion:forward(output, targets)
              -- estimate df/dW
              local df_do = self.criterion:backward(output, targets)
              self.module:backward(inputs, df_do)
              -- update evaluation counter
              self.evalCounter = self.evalCounter + inputs:size(1)
           end

           -- update evaluation counter
           self.batchCounter = self.batchCounter + 1

           -- verbose
           if self.verbose >= 2 then
              print('<BatchOptimization> ' .. self.batchCounter .. 'th batch took ' .. (sys.clock() - _t_) .. ' sec')
           end
           return self.output
        end

   -- (2) optimization callback
   if self.optimize then
      self:optimize(inputs, targets)
   end

   -- (3) update sample counter
   self.sampleCounter = self.sampleCounter + batchsize

   -- (4) return current output after optimization
   return self.output
end

function Batch:forward_mapreduce(inputs, targets, options)
   -- parameters
   local P = self.P

   -- transmit user hooks, if defined
   if not self.hooksets then
      parallel.children:send(self.prehook or '')
      parallel.children:send(self.posthook or '')
      self.hooksets = true
   end

   -- (0a) replicate output and gradParameters
   local outputsPartial = {}
   local gradParametersPartial = {}

   if self.copyBatch then
      -- (0) send same mini-batch to all workers
      for t = 1,P do
         self.children[t]:join()
         self.children[t]:send(inputs)
         self.children[t]:send(targets)
         self.children[t]:send(options)
      end
   else
      -- (0b) divide input/target batch into N batches, based on speed
      -- of each worker
      local inputss = {}
      local targetss = {}
      local optionss = {}
      local speed = 0
      for t = 1,P do
         speed = speed + self.children[t].speed
      end
      local n = 1
      for t = 1,P do
         inputss[t] = {}
         targetss[t] = {}
         optionss[t] = {}
         for i = 1,math.ceil(self.children[t].speed*(#inputs)/speed) do
            table.insert(inputss[t], inputs[n])
            table.insert(targetss[t], targets[n])
            if options then table.insert(optionss[t], options[n]) end
            n = n + 1
            if n > #inputs then break end
         end
      end

      -- (0c) send parts of mini-batch to each worker
      for t = 1,P do
         self.children[t]:join()
         self.children[t]:send(inputss[t])
         self.children[t]:send(targetss[t])
         self.children[t]:send(optionss[t])
      end
   end

   -- (1) construct a closure that compute f(inputs) + df/dW
   --     after each call to that function:
   --       + self.parameters contains the current X vector
   --       + self.gradParameters contains the estimated dF/dX vector
   --       + self.output contains the estimated (average) F(X)
   self.evaluate
      = function()
           -- verbose
           if self.verbose >= 2 then
              print('<BatchOptimization> evaluating f(X) + df/dX')
           end
           local _t_ = sys.clock()
           -- do map/reduce
           self.evaluate_map()
           self.evaluate_reduce()
           -- update evaluation counter
           self.evalCounter = self.evalCounter + 1
           -- verbose
           if self.verbose >= 2 then
              print('<BatchOptimization> ' .. self.evalCounter .. 'th evaluation took ' .. (sys.clock() - _t_) .. ' sec')
           end
           return self.output
        end

   -- (1a) the map part of the evaluation: compute partial gradients
   --      in separate threads
   self.evaluate_map
      = function()
           if self.map_hook then
              self:map_hook()
           else
              -- transmit new parameters to all workers
              self.children:join()
              self.children:send(self.parameters)
              -- then wait for all workers to return their partial gradParameters + outputs
              gradParametersPartial = self.children:receive()
              outputsPartial = self.children:receive()
              -- force cleanup
              collectgarbage()
           end
        end
   -- (1b) the reduce part of the evaluation: accumulate all
   --      partial estimates of the gradients
   self.evaluate_reduce
      = function()
           if self.reduce_hook then
              self:reduce_hook()
           else
              -- standard reduce is to sum the gradients
              -- accumulate partial gradients, and average
              self.gradParameters:zero()
              for t = 1,P do
                 self.gradParameters:add(gradParametersPartial[t])
              end
              self.gradParameters:div(#inputs)
              -- return average f(X)
              self.output = 0
              for t = 1,P do
                 self.output = self.output + outputsPartial[t]
              end
              self.output = self.output/#inputs
           end
        end

   if self.optimize then
      -- (2) optimization callback
      self:optimize()

      -- (3) reset workers so they're ready for next mini-batch
      -- only do this when we have an optimization hook
      self.children:join('break')
   end

   -- (4) update sample counter
   self.sampleCounter = self.sampleCounter + #inputs

   -- (5) return current output after optimization
   return self.output
end

-- [MS] this default worker code is too detailed needs to be a
-- skeleton which is easier to adapt... for now I am overriding this
-- whole function with the 2 closures in GenSGD

function Batch:setup_mapreduce ()
   -- (0) startup parallel package
   if not xrequire 'parallel' then
      xerror('install parallel for Lua to enable parallel computing (luarocks install parallel)',
             'nn.BatchOptimization')
   end

   -- (1) define code for workers
   local worker_code =
      function()
         -- require packages
         require 'nnx'

         -- retrieve optional code to setup worker
         precode = parallel.parent:receive()
         if type(precode) == 'function' then precode() end

         -- retrieve module + criterion at startup
         parallel.yield()
         module = parallel.parent:receive()
         criterion = parallel.parent:receive()

         -- create fake optimizer, for hooks
         optimizer = {module=module, criterion=criterion}

         -- retrieve optional prehook/posthook
         prehook = parallel.parent:receive()
         posthook = parallel.parent:receive()
         if type(prehook) ~= 'function' then prehook = nil end
         if type(posthook) ~= 'function' then posthook = nil end

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

               -- reset gradients
               gradParameters:zero()
               -- f is the average of all criterions
               local f_x = 0
               -- evaluate gradients on inputs for this thread
               for i = 1,#inputs do
                  -- user hook
                  if prehook then
                     prehook(optimizer, {inputs[i], targets[i], options[i]})
                  end
                  -- estimate f
                  local output = module:forward(inputs[i])
                  local err = criterion:forward(output, targets[i])
                  f_x = f_x + err
                  -- estimate df/dW
                  local df_do = criterion:backward(output, targets[i])
                  module:backward(inputs[i], df_do)
                  -- user hook
                  if posthook then
                     posthook(optimizer, {inputs[i], targets[i], options[i]})
                  end
               end
               -- now send back gradParameters + partial output
               parallel.parent:send(gradParameters)
               parallel.parent:send(f_x)
               -- force cleanup
               collectgarbage()
            end
         end
      end
   -- (2) dispatch workers
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
                 end

   local ok,err = pcall(setup)
   if not ok then parallel.close() error(err) end
end
