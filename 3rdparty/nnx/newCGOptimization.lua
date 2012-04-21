local CG,parent = torch.class('nn.newCGOptimization', 'nn.BatchOptimization')
--
-- wrapper around Koray's cg function which implements rasmussen's
-- matlab cg in pure lua.
-- Author: Marco Scoffier
--
function CG:__init(...)
   parent.__init(self, ...)
   xlua.unpack_class(self, {...},
                     'cgOptimization', nil,
                     {arg='rho',    type='number', default=0.1},
                     {arg='sig',    type='number', default=0.5},
                     {arg='int',    type='number', default=0.1},
                     {arg='ext',    type='number', default=3.0},
                     {arg='max',    type='number', default=20},
                     {arg='ratio',  type='number', default=100},
                     {arg='length', type='number', default=25},
                     {arg='red',    type='number', default=1},
                     {arg='verbose', type='number', default=0}
                  )



   -- we need three points for the interpolation/extrapolation stuff
   self.df1, self.df2, self.df3 = torch.Tensor(),torch.Tensor(),torch.Tensor()

   self.df1:resizeAs(self.parameters)
   self.df2:resizeAs(self.parameters)
   self.df3:resizeAs(self.parameters)

   -- search direction
   self.s = torch.Tensor():resizeAs(self.parameters)

   -- we need a temp storage for X
   self.x0 = torch.Tensor():resizeAs(self.parameters)
   self.df0 = torch.Tensor():resizeAs(self.parameters)

end

function CG:optimize()
   local rho = self.rho
   local sig = self.sig
   local int = self.int
   local ext = self.ext
   local max = self.max
   local ratio = self.ratio
   local length = self.length
   local red = self.red
   local verbose = self.verbose

   local i = 0
   local ls_failed = 0
   local fx  = {}
   
   -- we need three points for the interpolation/extrapolation stuff
   local z1,  z2,  z3 = 0,0,0
   local d1,  d2,  d3 = 0,0,0
   local f1,  f2,  f3 = 0,0,0
   local df1,df2,df3 = self.df1, self.df2, self.df3

   local x = self.parameters
   local s = self.s

   local x0 = self.x0
   local f0 = 0
   local df0 = self.df0

   -- the magic function: will update the parameter vector using CG
   -- evaluate at initial point
   f1 = self.evaluate()
   df1:copy(self.gradParameters)
   i=i+1

   -- initial search direction
   s:copy(df1):mul(-1)

   d1 = -s:dot(s )         -- slope
   z1 = red/(1-d1)         -- initial step

   while i < math.abs(length) do

      x0:copy(x)
      f0 = f1
      df0:copy(df1)
      x:add(z1,s)
      f2 = self.evaluate()
      df2:copy(self.gradParameters)
      i=i+1
      d2 = df2:dot(s)
      f3,d3,z3 = f1,d1,-z1   -- init point 3 equal to point 1
      local m = math.min(max,length-i)
      local success = 0
      local limit = -1
      
      while true do
	 while (f2 > f1+z1*rho*d1 or d2 > -sig*d1) and m > 0 do
	    limit = z1
	    if f2 > f1 then
	       z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)
	    else
	       local A = 6*(f2-f3)/z3+3*(d2+d3)
	       local B = 3*(f3-f2)-z3*(d3+2*d2)
	       z2 = (math.sqrt(B*B-A*d2*z3*z3)-B)/A
	    end
	    if z2 ~= z2 or z2 == math.huge or z2 == -math.huge then
	       z2 = z3/2;
	    end
	    z2 = math.max(math.min(z2, int*z3),(1-int)*z3);
	    z1 = z1 + z2;
	    x:add(z2,s)
	    f2 = self.evaluate()
	    df2:copy(self.gradParameters)
	    i=i+1
	    m = m - 1
	    d2 = df2:dot(s)
	    z3 = z3-z2;
	 end
	 if f2 > f1+z1*rho*d1 or d2 > -sig*d1 then
	    break
	 elseif d2 > sig*d1 then
	    success = 1;
	    break;
	 elseif m == 0 then
	    break;
	 end
	 local A = 6*(f2-f3)/z3+3*(d2+d3);
	 local B = 3*(f3-f2)-z3*(d3+2*d2);
	 z2 = -d2*z3*z3/(B+math.sqrt(B*B-A*d2*z3*z3))

	 if z2 ~= z2 or z2 == math.huge or z2 == -math.huge or z2 < 0 then
	    if limit < -0.5 then
	       z2 = z1 * (ext -1)
	    else
	       z2 = (limit-z1)/2
	    end
	 elseif (limit > -0.5) and (z2+z1) > limit then
	    z2 = (limit-z1)/2
	 elseif limit < -0.5 and (z2+z1) > z1*ext then
	    z2 = z1*(ext-1)
	 elseif z2 < -z3*int then
	    z2 = -z3*int
	 elseif limit > -0.5 and z2 < (limit-z1)*(1-int) then
	    z2 = (limit-z1)*(1-int)
	 end
	 f3=f2; d3=d2; z3=-z2;
	 z1 = z1+z2;

	 x:add(z2,s)

	 f2 = self.evaluate()
	 df2:copy(self.gradParameters)
	 i=i+1
	 m = m - 1
	 d2 = df2:dot(s)
      end
      if success == 1 then
	 f1 = f2
	 fx[#fx+1] = f1;
	 local ss = (df2:dot(df2)-df2:dot(df1)) / df1:dot(df1)
	 s:mul(ss)
	 s:add(-1,df2)
	 local tmp = df1:clone()
	 df1:copy(df2)
	 df2:copy(tmp)
	 d2 = df1:dot(s)
	 if d2> 0 then
	    s:copy(df1)
	    s:mul(-1)
	    d2 = -s:dot(s)
	 end

	 z1 = z1 * math.min(ratio, d1/(d2-1e-320))
	 d1 = d2
	 ls_failed = 0
      else
	 x:copy(x0)
	 f1 = f0
	 df1:copy(df0)
	 if ls_failed or i>length then
	    break
	 end	 
	 local tmp = df1:clone()
	 df1:copy(df2)
	 df2:copy(tmp)
	 s:copy(df1)
	 s:mul(-1)
	 d1 = -s:dot(s)
	 z1 = 1/(1-d1)
	 ls_failed = 1
      end
   end
   self.output = f1 -- self.evaluate(x)
   collectgarbage()
end
