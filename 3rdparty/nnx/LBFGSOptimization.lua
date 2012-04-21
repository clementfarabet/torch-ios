local LBFGS,parent = torch.class('nn.LBFGSOptimization', 'nn.BatchOptimization')

function LBFGS:__init(...)
   require 'liblbfgs'
   parent.__init(self, ...)
   xlua.unpack_class(self, {...},
                     'LBFGSOptimization', nil,
                     {arg='maxEvaluation', type='number',
                      help='maximum nb of function evaluations per pass (0 = no max)', default=0},
                     {arg='maxIterations', type='number',
                      help='maximum nb of iterations per pass (0 = no max)', default=0},
                     {arg='maxLineSearch', type='number',
                      help='maximum nb of steps in line search', default=20},
                     {arg='sparsity', type='number',
                      help='sparsity coef (Orthantwise C)', default=0},
                     {arg='linesearch', type='string',
                      help='type of linesearch used: morethuente, armijo, wolfe, strong_wolfe',
                      default='wolfe'},
                     {arg='parallelize', type='number',
                      help='parallelize onto N cores (experimental!)', default=1}
                  )
   local linesearch = 2
   if not (self.linesearch == 'wolfe') then
      if self.linesearch == 'morethuente' then
         linesearch = 0
      elseif self.linesearch == 'armijo' then
         linesearch = 1
      elseif self.linesearch == 'strong_wolfe' then
         linesearch = 3
      end
   end
   -- init LBFGS state
   lbfgs.init(self.parameters, self.gradParameters,
              self.maxEvaluation, self.maxIterations, self.maxLineSearch,
              self.sparsity, linesearch, self.verbose)
end

function LBFGS:optimize()
   -- callback for lBFGS
   lbfgs.evaluate = function()
                       local loss = self.evaluate()
                       if self.allreduce then
                          local losst = torch.Tensor(1):fill(loss)
                          allreduce.accumulate(losst)
                          allreduce.accumulate(self.gradParameters)
                          loss = losst[1]
                       end
                       return loss
                    end

   -- the magic function: will update the parameter vector according to the l-BFGS method
   self.output = lbfgs.run()
end
