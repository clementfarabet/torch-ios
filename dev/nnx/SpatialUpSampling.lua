local SpatialUpSampling, parent = torch.class('nn.SpatialUpSampling', 'nn.Module')

local help_desc = [[
Applies a 2D up-sampling over an input image composed of
several input planes. The input tensor in forward(input) is
expected to be a 3D tensor (nInputPlane x width x height).
The number of output planes will be the same as nInputPlane.

The upsampling is done using the simple nearest neighbor
technique. For interpolated (bicubic) upsampling, use 
nn.SpatialReSampling().

If the input image is a 3D tensor nInputPlane x width x height,
the output image size will be nInputPlane x owidth x oheight where

owidth  = width*dW
oheight  = height*dH ]]

function SpatialUpSampling:__init(...)
   parent.__init(self)

   -- get args
   xlua.unpack_class(self, {...}, 'nn.SpatialUpSampling',  help_desc,
                     {arg='dW', type='number', help='stride width', req=true},
                     {arg='dH', type='number', help='stride height', req=true})
end

function SpatialUpSampling:updateOutput(input)
   self.output:resize(input:size(1), input:size(2) * self.dH, input:size(3) * self.dW)
   input.nn.SpatialUpSampling_updateOutput(self, input)
   return self.output
end

function SpatialUpSampling:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.SpatialUpSampling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
