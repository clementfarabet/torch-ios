local BatchTrainer, parent = torch.class('nn.BatchTrainer', 'nn.OnlineTrainer')

-- Essentially simialar to the OnlineTrainer but only used the parts
-- of the code which prepare the data and the tester. train() has been
-- replaced by nextBatch() which moves the trainer one batch further
-- in the data.  When the first epoch is finished then the batches are
-- reused.  Each call to optimizer.forward() in nextBatch() creates a
-- closure with the current batch as input.

function BatchTrainer:__init(...)
   local args = {...}
   parent.__init(self, args)
   -- unpack args
   xlua.unpack_class(
      self, args,
      'BatchTrainer', 
      'A modified version of the general-purpose online trainer class.\n'
	 .. ' which only preps the input batch and calls optimizer to\n'
	 .. ' create a closure\n',
      {arg='trainset', type='nn.DataList', 
       help='dataset from which to draw batches', req=true},
      {arg='module', type='nn.Module', help='a module to train', req=true},
      {arg='criterion', type='nn.Criterion', 
       help='a criterion to estimate the error'},
      {arg='preprocessor', type='nn.Module', 
       help='a preprocessor to prime the data before the module'},
      {arg='optimizer', type='nn.Optimization', 
       help='an optimization method'}, 
      {arg='batchSize', type='number', 
       help='[mini] batch size', default=1},
      {arg='maxEpoch', type='number', 
       help='maximum number of epochs', default=50},
      {arg='dispProgress', type='boolean', 
       help='display a progress bar during training/testing', default=true},
      {arg='save', type='string', 
       help='path to save networks and log training'},
      {arg='timestamp', type='boolean', 
       help='if true, appends a timestamp to each network saved', default=false}
   )
   self.epoch = 1
   self.batch = nil
   self.trainOffset = nil
end

-- update the counters
function BatchTrainer:next()
   if not self.batch or not self.trainOffset then
      -- initialize
      self.batch = 1
      self.trainOffset = 1
   else
      -- hook to run something on the current batch
      -- (for eg. if you want to run a test on this batch before
      -- switching to the next)
      if self.hookTrainBatch then
	 self.hookTrainBatch(self)
      end

      -- simple batch increment
      self.batch = self.batch + 1
      self.trainOffset = self.trainOffset + self.batchSize
      
      -- test for new epoch
      if self.trainOffset > self.trainset:size() then

	 -- hook to run on current epoch before switching to next
	 if self.hookTrainEpoch then
	    self.hookTrainEpoch(self)
	 end

	 if self.save then self:log() end

	 self.trainOffset = 1
	 self.epoch = self.epoch + 1
	 self.batch = 1
      end
      
      -- on all but the first batch we need to reset the children
      if optimizer.parallelize > 1 then 
	 parallel.children:send('break')
      end

   end
   -- disp progress
   if self.dispProgress then
      xlua.progress(self.trainOffset, self.trainset:size())
   end

end

-- this function is called train() in the online trainer.  I seems to
-- make more sense to call it next_batch() here as the training is
-- done outside of this code.

function BatchTrainer:nextBatch()
   self:next()
   local module = self.module
   local criterion = self.criterion
   local t = self.trainOffset
   local ds = self.trainset:size()
   local bs = self.batchSize
   
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. self.epoch 
	 .. ' batch # '..self.batch
	 .. ' [batchSize = ' .. self.batchSize .. ']')

   -- create mini batch
   self.inputs = self.inputs or {}
   self.targets = self.targets or {}
   local inputs = {}
   local targets = {}
   if not self.inputs[self.batch] then

      self.inputs[self.batch] = {}
      inputs = self.inputs[self.batch] 
      self.targets[self.batch] = {}
      targets = self.targets[self.batch]

      for i = t,math.min(t+bs-1,ds) do
	 -- load new sample
	 local sample = self.trainset[i]
	 local input = sample[1]
	 local target = sample[2]
	 
	 -- optional preprocess (no learning is done for that guy)
	 if self.preprocessor then input = self.preprocessor:forward(input) end
	 
	 -- store input/target
	 table.insert(inputs, input)
	 table.insert(targets, target)
      end
   else  
      -- get batch from cache
      inputs = self.inputs[self.batch] 
      targets = self.targets[self.batch]
   end   

   -- set up closure batch.evaluate() for optimizer
   local error = self.optimizer:forward(inputs, targets)
   
end

-- special test to just get results of current batch
function BatchTrainer:testBatch()
   local criterion = self.criterion
   local module = self.module
   
   local inputs = self.inputs[self.batch]
   local targets = self.targets[self.batch]
   
   self.currentError = 0
   
   for i = 1,#inputs do 
      local input = inputs[i]
      local target = targets[i]
      if criterion then
         self.currentError = self.currentError + 
	    criterion:forward(module:forward(input), target)
      else
         local _,error = module:forward(input, target)
         self.currentError = self.currentError + error
      end
      -- user hook
      if self.hookTestSample then
         self.hookTestSample(self, {input, target})
      end
   end
end

