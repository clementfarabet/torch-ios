local Optimization = torch.class('nn.Optimization')

function Optimization:__init()
   self.output = 0
end

function Optimization:forward(inputs, targets)
   self.output = 0
   print('<Optimization> WARNING: this is a virtual function, please overload !')
   return self.output
end
