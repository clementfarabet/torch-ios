----------------------------------------------------------------------
--
-- Copyright (c) 2011 Clement Farabet, Marco Scoffier, 
--                    Koray Kavukcuoglu, Benoit Corda
--
-- 
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
-- 
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- 
----------------------------------------------------------------------
-- description:
--     xlua - lots of new trainable modules that extend the nn 
--            package.
--
-- history: 
--     July  5, 2011, 8:51PM - import from Torch5 - Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'nn'

-- create global nnx table:
nnx = {}

-- c lib:
require 'libnnx'

-- for testing:
torch.include('nnx', 'test-all.lua')
torch.include('nnx', 'test-omp.lua')

-- tools:
torch.include('nnx', 'ConfusionMatrix.lua')
torch.include('nnx', 'Logger.lua')
torch.include('nnx', 'Probe.lua')

-- OpenMP module:
torch.include('nnx', 'OmpModule.lua')

-- pointwise modules:
torch.include('nnx', 'Minus.lua')

-- spatial (images) operators:
torch.include('nnx', 'SpatialLinear.lua')
torch.include('nnx', 'SpatialClassifier.lua')
torch.include('nnx', 'SpatialPadding.lua')
torch.include('nnx', 'SpatialNormalization.lua')
torch.include('nnx', 'SpatialUpSampling.lua')
torch.include('nnx', 'SpatialDownSampling.lua')
torch.include('nnx', 'SpatialReSampling.lua')
torch.include('nnx', 'SpatialRecursiveFovea.lua')
torch.include('nnx', 'SpatialFovea.lua')
torch.include('nnx', 'SpatialPyramid.lua')
torch.include('nnx', 'SpatialGraph.lua')
torch.include('nnx', 'SpatialMatching.lua')
torch.include('nnx', 'SpatialMaxSampling.lua')
torch.include('nnx', 'SpatialColorTransform.lua')
torch.include('nnx', 'SpatialConvolutionSparse.lua')

-- criterions:
torch.include('nnx', 'SuperCriterion.lua')
torch.include('nnx', 'SparseCriterion.lua')
torch.include('nnx', 'DistNLLCriterion.lua')
torch.include('nnx', 'KLDivCriterion.lua')
torch.include('nnx', 'DistMarginCriterion.lua')
torch.include('nnx', 'SpatialMSECriterion.lua')
torch.include('nnx', 'SpatialClassNLLCriterion.lua')
torch.include('nnx', 'SpatialSparseCriterion.lua')

-- optimizations:
torch.include('nnx', 'Optimization.lua')
torch.include('nnx', 'BatchOptimization.lua')
torch.include('nnx', 'SNESOptimization.lua')
torch.include('nnx', 'SGDOptimization.lua')
torch.include('nnx', 'ASGDOptimization.lua')
torch.include('nnx', 'LBFGSOptimization.lua')
torch.include('nnx', 'CGOptimization.lua')
torch.include('nnx', 'newCGOptimization.lua')
torch.include('nnx', 'GeneticSGDOptimization.lua')
torch.include('nnx', 'DiagHessian.lua')

-- trainers:
torch.include('nnx', 'Trainer.lua')
torch.include('nnx', 'OnlineTrainer.lua')
torch.include('nnx', 'BatchTrainer.lua')

-- conversion helper:
torch.include('nnx', 'Type.lua')

-- datasets:
torch.include('nnx', 'DataSet.lua')
torch.include('nnx', 'DataList.lua')
torch.include('nnx', 'DataSetLabelMe.lua')

-- nn helpers:
function nnx.empty(module)
   if module.modules then
      -- find submodules in classic containers 'modules'
      for _,module in ipairs(module.modules) do
         nnx.empty(module)
      end
   else
      -- find arbitrary submodules
      for k,entry in pairs(module) do
         local type = torch.typename(entry)
         if type and type:find('^nn.') then
            nnx.empty(entry)
         elseif type(entry) == 'table' then
            for i,entry in ipairs(entry) do
               local type = torch.typename(entry)
               if type and type:find('^nn.') then
                  nnx.empty(entry)
               end
            end
         end
      end
   end
   -- empty module
   if module.output and module.output.resize then 
      module.output:resize()
      module.output:storage():resize(0)
   end
   if module.gradInput and module.gradInput.resize then 
      module.gradInput:resize()
      module.gradInput:storage():resize(0)
   end
end

local function get(module, holder, params)
   -- find submodules in classic containers 'modules'
   if module.modules then
      for _,module in ipairs(module.modules) do
         get(module, holder, params)
      end
   else
      -- find parameters and store them
      for _,param in ipairs(params) do
         if module[param] then
            table.insert(holder, module[param])
         end
      end
   end
end

function nnx.getParameters(...)
   -- to hold all parameters found
   local holder = {}
   -- call recursive call
   local modules = {...}
   for _,module in ipairs(modules) do
      get(module, holder, {'weight', 'bias'})
   end
   -- return all parameters found
   return holder
end

function nnx.getGradParameters(...)
   -- to hold all parameters found
   local holder = {}
   -- call recursive call
   local modules = {...}
   for _,module in ipairs(modules) do
      get(module, holder, {'gradWeight', 'gradBias'})
   end
   -- return all parameters found
   return holder
end

function nnx.getDiagHessianParameters(...)
   -- to hold all parameters found
   local holder = {}
   -- call recursive call
   local modules = {...}
   for _,module in ipairs(modules) do
      get(module, holder, {'diagHessianWeight', 'diagHessianBias'})
   end
   -- return all parameters found
   return holder
end

function nnx.flattenParameters(parameters)
   -- already flat ?
   local flat = true
   for k = 2,#parameters do
      if parameters[k]:storage() ~= parameters[k-1]:storage() then
         flat = false
         break
      end
   end
   if flat then
      local nParameters = 0
      for k,param in ipairs(parameters) do
         nParameters = nParameters + param:nElement()
      end
      flatParameters = parameters[1].new(parameters[1]:storage())
      if nParameters ~= flatParameters:nElement() then
         error('<nnx.flattenParameters> weird parameters')
      end
      return flatParameters
   end
   -- compute offsets of each parameter
   local offsets = {}
   local sizes = {}
   local strides = {}
   local elements = {}
   local storageOffsets = {}
   local params = {}
   local nParameters = 0
   for k,param in ipairs(parameters) do
      table.insert(offsets, nParameters+1)
      table.insert(sizes, param:size())
      table.insert(strides, param:stride())
      table.insert(elements, param:nElement())
      table.insert(storageOffsets, param:storageOffset())
      local isView = false
      for i = 1,k-1 do
         if param:storage() == parameters[i]:storage() then
            offsets[k] = offsets[i]
            if storageOffsets[k] ~= storageOffsets[i] or elements[k] ~= elements[i] then
               error('<nnx.flattenParameters> cannot flatten shared weights with different structures')
            end
            isView = true
            break
         end
      end
      if not isView then
         nParameters = nParameters + param:nElement()
      end
   end
   -- create flat vector
   local flatParameters = parameters[1].new(nParameters)
   local storage = flatParameters:storage()
   -- reallocate all parameters in flat vector
   for i = 1,#parameters do
      local data = parameters[i]:clone()
      parameters[i]:set(storage, offsets[i], elements[i]):resize(sizes[i],strides[i]):copy(data)
      data = nil
      collectgarbage()
   end
   -- cleanup
   collectgarbage()
   -- return new flat vector that contains all discrete parameters
   return flatParameters
end
