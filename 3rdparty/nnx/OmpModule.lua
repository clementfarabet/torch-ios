local OmpModule,parent = torch.class('nn.OmpModule','nn.Module')

function OmpModule:__init()
   parent.__init(self)
   if openmp then
      self.nThread = openmp.getNumThreads()
   else
      self.nThread = 1
   end
end

function OmpModule:read(file)
   parent.read(self, file)
   if openmp then
      self.nThread = openmp.getNumThreads()
   else
      self.nThread = 1
   end
end
