local SpatialConvolutionSparse, parent = torch.class('nn.SpatialConvolutionSparse', 
                                                     'nn.SpatialConvolutionMap')

function SpatialConvolutionSparse:__init(...)
   parent.__init(self,...)
end
