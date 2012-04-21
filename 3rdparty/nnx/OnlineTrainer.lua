local OnlineTrainer, parent = torch.class('nn.OnlineTrainer','nn.Trainer')

function OnlineTrainer:__init(...)
   parent.__init(self)
   -- unpack args
   xlua.unpack_class(self, {...},
      'OnlineTrainer', 

      'A general-purpose online trainer class.\n'
         .. 'Provides 4 user hooks to perform extra work after each sample, or each epoch:\n'
         .. '> trainer = nn.OnlineTrainer(...) \n'
         .. '> trainer.hookTrainSample = function(trainer, sample) ... end \n'
         .. '> trainer.hookTrainEpoch = function(trainer) ... end \n'
         .. '> trainer.hookTestSample = function(trainer, sample) ... end \n'
         .. '> trainer.hookTestEpoch = function(trainer) ... end \n'
         .. '> ',

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
end

function OnlineTrainer:log()
   -- save network
   local filename = self.save
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if self.timestamp then
      -- use a timestamp to store all networks uniquely
      filename = filename .. '-' .. os.date("%Y_%m_%d_%X")
   else
      -- if no timestamp, just store the previous one
      if sys.filep(filename) then
         os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
      end
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, self.module)
end

function OnlineTrainer:train(dataset)
   self.epoch = self.epoch or 1
   local module = self.module
   local criterion = self.criterion
   self.trainset = dataset

   while true do
      print('<trainer> on training set:')
      print("<trainer> online epoch # " .. self.epoch .. ' [batchSize = ' .. self.batchSize .. ']')

      self.time = sys.clock()
      self.currentError = 0
      for t = 1,dataset:size(),self.batchSize do
         -- disp progress
         if self.dispProgress then
            xlua.progress(t, dataset:size())
         end

         -- create mini batch
         local inputs = {}
         local targets = {}
         for i = t,math.min(t+self.batchSize-1,dataset:size()) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]
            local target = sample[2]

            -- optional preprocess (no learning is done for that guy)
            if self.preprocessor then input = self.preprocessor:forward(input) end

            -- store input/target
            table.insert(inputs, input)
            table.insert(targets, target)
         end

         -- optimize the model given current input/target set
         local error = self.optimizer:forward(inputs, targets)

         -- accumulate error
         self.currentError = self.currentError + error

         -- call user hook, if any
         if self.hookTrainSample then
            self.hookTrainSample(self, {inputs[#inputs], targets[#targets]})
         end
      end

      self.currentError = self.currentError / dataset:size()
      print("<trainer> current error = " .. self.currentError)

      self.time = sys.clock() - self.time
      self.time = self.time / dataset:size()
      print("<trainer> time to learn 1 sample = " .. (self.time*1000) .. 'ms')

      if self.hookTrainEpoch then
         self.hookTrainEpoch(self)
      end

      if self.save then self:log() end

      self.epoch = self.epoch + 1

      if self.maxEpoch > 0 and self.epoch > self.maxEpoch then
         print("<trainer> you have reached the maximum number of epochs")
         break
      end
   end
end


function OnlineTrainer:test(dataset)
   print('<trainer> on testing Set:')

   local module = self.module
   local criterion = self.criterion
   self.currentError = 0
   self.testset = dataset

   self.time = sys.clock()
   for t = 1,dataset:size() do
      -- disp progress
      if self.dispProgress then
         xlua.progress(t, dataset:size())
      end

      -- get new sample
      local sample = dataset[t]
      local input = sample[1]
      local target = sample[2]
      
      -- test sample through current model
      if self.preprocessor then input = self.preprocessor:forward(input) end
      if criterion then
         self.currentError = self.currentError + 
	    criterion:forward(module:forward(input), target)
      else
         local _,error = module:forward(input, target)
         self.currentError = self.currentError + error
      end

      -- user hook
      if self.hookTestSample then
         self.hookTestSample(self, sample)
      end
   end

   self.currentError = self.currentError / dataset:size()
   print("<trainer> test current error = " .. self.currentError)

   self.time = sys.clock() - self.time
   self.time = self.time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (self.time*1000) .. 'ms')

   if self.hookTestEpoch then
      self.hookTestEpoch(self)
   end

   return self.currentError
end
