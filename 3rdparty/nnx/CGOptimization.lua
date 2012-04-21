local CG,parent = torch.class('nn.CGOptimization', 'nn.BatchOptimization')

function CG:__init(...)
   require 'liblbfgs'
   parent.__init(self, ...)
   xlua.unpack_class(self, {...},
                     'CGOptimization', nil,
                     {arg='maxEvaluation', type='number',
                      help='maximum nb of function evaluations per pass (0 = no max)', default=0},
                     {arg='maxIterations', type='number',
                      help='maximum nb of iterations per pass (0 = no max)', default=0},
                     {arg='maxLineSearch', type='number',
                      help='maximum nb of steps in line search', default=20},
                     {arg='sparsity', type='number',
                      help='sparsity coef (Orthantwise C)', default=0},
                     {arg='linesearch', type='string',
                      help=[[ type of linesearch used: 
                            "morethuente", "m",
                            "armijo", "a",
                            "wolfe", "w",
                            "strong_wolfe", "s"
                      ]],
                      default='wolfe'},
                     {arg='momentum', type='string',
                      help=[[ type of momentum used: 
                            "fletcher-reeves", "fr",
                            "polack-ribiere", "pr",
                            "hestens-steifel", "hs",
                            "gilbert-nocedal", "gn"
                      ]],
                      default='fletcher-reeves'},
                     {arg='parallelize', type='number',
                      help='parallelize onto N cores (experimental!)', default=1}
                  )
   local linesearch = 2
   if not (self.linesearch == 'w' or self.linesearch == 'wolfe') then
      if self.linesearch == 'm' or self.linesearch == 'morethuente' then
         linesearch = 0
      elseif self.linesearch == 'a' or self.linesearch == 'armijo' then
         linesearch = 1
      elseif self.linesearch == 's' or self.linesearch == 'strong_wolfe' then
         linesearch = 3
      end
   end

   local momentum = 0
   if not (self.momentum == 'fr' or self.momentum == 'fletcher-reeves') then
      if self.momentum == 'pr' or self.momentum == 'polack-ribiere' then
         momentum = 1
      elseif self.momentum == 'hs' or self.momentum == 'hestens-steifel' then
         momentum = 2
      elseif self.momentum == 'gn' or self.momentum == 'gilbert-nocedal' then
         momentum = 3
      end
   end

   -- init CG state
   cg.init(self.parameters, self.gradParameters,
           self.maxEvaluation, self.maxIterations, self.maxLineSearch,
           momentum, linesearch, self.verbose)
end

function CG:optimize()
   -- callback for lBFGS
   lbfgs.evaluate = self.evaluate
   -- the magic function: will update the parameter vector using CG
   self.output = cg.run()
end
