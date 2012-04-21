/*
 *      Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* $Id$ */

/*
  This library is a C port of the FORTRAN implementation of Limited-memory
  Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
  The original FORTRAN source code is available at:
  http://www.ece.northwestern.edu/~nocedal/lbfgs.html

  The L-BFGS algorithm is described in:
  - Jorge Nocedal.
  Updating Quasi-Newton Matrices with Limited Storage.
  <i>Mathematics of Computation</i>, Vol. 35, No. 151, pp. 773--782, 1980.
  - Dong C. Liu and Jorge Nocedal.
  On the limited memory BFGS method for large scale optimization.
  <i>Mathematical Programming</i> B, Vol. 45, No. 3, pp. 503-528, 1989.

  The line search algorithms used in this implementation are described in:
  - John E. Dennis and Robert B. Schnabel.
  <i>Numerical Methods for Unconstrained Optimization and Nonlinear
  Equations</i>, Englewood Cliffs, 1983.
  - Jorge J. More and David J. Thuente.
  Line search algorithm with guaranteed sufficient decrease.
  <i>ACM Transactions on Mathematical Software (TOMS)</i>, Vol. 20, No. 3,
  pp. 286-307, 1994.

  This library also implements Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
  method presented in:
  - Galen Andrew and Jianfeng Gao.
  Scalable training of L1-regularized log-linear models.
  In <i>Proceedings of the 24th International Conference on Machine
  Learning (ICML 2007)</i>, pp. 33-40, 2007.

  I would like to thank the original author, Jorge Nocedal, who has been
  distributing the effieicnt and explanatory implementation in an open source
  licence.
*/

#ifdef  HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef WITH_CUDA
#include <THC/THC.h>
#endif

#include "TH.h"
#include "luaT.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <lbfgs.h>

#include "lbfgs_ansi.h"

#define min2(a, b)      ((a) <= (b) ? (a) : (b))
#define max2(a, b)      ((a) >= (b) ? (a) : (b))
#define max3(a, b, c)   max2(max2((a), (b)), (c));

/* extra globals: counters, verbose flag */
static int nEvaluation = 0;
static int nIteration  = 0;
static int verbose     = 0;

struct tag_callback_data {
  int n;
  void *instance;
  lbfgs_evaluate_t proc_evaluate;
  lbfgs_progress_t proc_progress;
};
typedef struct tag_callback_data callback_data_t;

struct tag_iteration_data {
  lbfgsfloatval_t alpha;
  lbfgsfloatval_t *s;     /* [n] */
  lbfgsfloatval_t *y;     /* [n] */
  lbfgsfloatval_t ys;     /* vecdot(y, s) */
};
typedef struct tag_iteration_data iteration_data_t;

static const lbfgs_parameter_t _def_param = {
  6,                          /* max nb or corrections stored, to estimate hessian */
  1e-5,                       /* espilon = stop condition on f(x) */
  0,                          /* - past */
  1e-5,                       /* - delta */
  0,                          /* number of complete iterations (0 = inf) */
  0,                          /* number of function evaluations (0 = inf) */
  1.0e-16,                    /* floating-point precision */
  LBFGS_LINESEARCH_DEFAULT,   /* line search method */
  40,                         /* max number of trials for line search */
  1e-20,                      /* min step for line search */
  1e20,                       /* max step for line search */
  1e-4,                       /* ftol = granularity for f(x) estimation */
  0.9,                        /* wolfe */
  0.9,                        /* gtol = granularity for df/dx estimation */
  0.0,                        /* sparsity constraint */
  0,                          /* sparsity offset */
  -1,                          /* sparsity end */
  CG_FLETCHER_REEVES,         /* momentum type */
};



/* Forward function declarations. */

typedef int (*line_search_proc)(
                                int n,
                                lbfgsfloatval_t *x,
                                lbfgsfloatval_t *f,
                                lbfgsfloatval_t *g,
                                lbfgsfloatval_t *s,
                                lbfgsfloatval_t *stp,
                                const lbfgsfloatval_t* xp,
                                const lbfgsfloatval_t* gp,
                                lbfgsfloatval_t *wa,
                                callback_data_t *cd,
                                const lbfgs_parameter_t *param
                                );

static int line_search_backtracking(
                                    int n,
                                    lbfgsfloatval_t *x,
                                    lbfgsfloatval_t *f,
                                    lbfgsfloatval_t *g,
                                    lbfgsfloatval_t *s,
                                    lbfgsfloatval_t *stp,
                                    const lbfgsfloatval_t* xp,
                                    const lbfgsfloatval_t* gp,
                                    lbfgsfloatval_t *wa,
                                    callback_data_t *cd,
                                    const lbfgs_parameter_t *param
                                    );

static int line_search_backtracking_owlqn(
                                          int n,
                                          lbfgsfloatval_t *x,
                                          lbfgsfloatval_t *f,
                                          lbfgsfloatval_t *g,
                                          lbfgsfloatval_t *s,
                                          lbfgsfloatval_t *stp,
                                          const lbfgsfloatval_t* xp,
                                          const lbfgsfloatval_t* gp,
                                          lbfgsfloatval_t *wp,
                                          callback_data_t *cd,
                                          const lbfgs_parameter_t *param
                                          );

static int line_search_morethuente(
                                   int n,
                                   lbfgsfloatval_t *x,
                                   lbfgsfloatval_t *f,
                                   lbfgsfloatval_t *g,
                                   lbfgsfloatval_t *s,
                                   lbfgsfloatval_t *stp,
                                   const lbfgsfloatval_t* xp,
                                   const lbfgsfloatval_t* gp,
                                   lbfgsfloatval_t *wa,
                                   callback_data_t *cd,
                                   const lbfgs_parameter_t *param
                                   );

static int update_trial_interval(
                                 lbfgsfloatval_t *x,
                                 lbfgsfloatval_t *fx,
                                 lbfgsfloatval_t *dx,
                                 lbfgsfloatval_t *y,
                                 lbfgsfloatval_t *fy,
                                 lbfgsfloatval_t *dy,
                                 lbfgsfloatval_t *t,
                                 lbfgsfloatval_t *ft,
                                 lbfgsfloatval_t *dt,
                                 const lbfgsfloatval_t tmin,
                                 const lbfgsfloatval_t tmax,
                                 int *brackt
                                 );

static lbfgsfloatval_t owlqn_x1norm(
                                    const lbfgsfloatval_t* x,
                                    const int start,
                                    const int n
                                    );

static void owlqn_pseudo_gradient(
                                  lbfgsfloatval_t* pg,
                                  const lbfgsfloatval_t* x,
                                  const lbfgsfloatval_t* g,
                                  const int n,
                                  const lbfgsfloatval_t c,
                                  const int start,
                                  const int end
                                  );

static void owlqn_project(
                          lbfgsfloatval_t* d,
                          const lbfgsfloatval_t* sign,
                          const int start,
                          const int end
                          );


#if     defined(USE_SSE) && (defined(__SSE__) || defined(__SSE2__))
static int round_out_variables(int n)
{
  n += 7;
  n /= 8;
  n *= 8;
  return n;
}
#endif/*defined(USE_SSE)*/

lbfgsfloatval_t* lbfgs_malloc(int n)
{
#if     defined(USE_SSE) && (defined(__SSE__) || defined(__SSE2__))
  n = round_out_variables(n);
#endif/*defined(USE_SSE)*/
  return (lbfgsfloatval_t*)vecalloc(sizeof(lbfgsfloatval_t) * n);
}

void lbfgs_free(lbfgsfloatval_t *x)
{
  vecfree(x);
}

void lbfgs_parameter_init(lbfgs_parameter_t *param)
{
  memcpy(param, &_def_param, sizeof(*param));
}

int check_params (int n, lbfgs_parameter_t param)
{
#if     defined(USE_SSE) && (defined(__SSE__) || defined(__SSE2__))
  /* Round out the number of variables. */
  n = round_out_variables(n);
#endif/*defined(USE_SSE)*/

  /* Check the input parameters for errors. */
  if (n <= 0) {
    return LBFGSERR_INVALID_N;
  }
#if     defined(USE_SSE) && (defined(__SSE__) || defined(__SSE2__))
  if (n % 8 != 0) {
    return LBFGSERR_INVALID_N_SSE;
  }
  if ((uintptr_t)(const void*)x % 16 != 0) {
    return LBFGSERR_INVALID_X_SSE;
  }
#endif/*defined(USE_SSE)*/
  if (param.epsilon < 0.) {
    return LBFGSERR_INVALID_EPSILON;
  }
  if (param.past < 0) {
    return LBFGSERR_INVALID_TESTPERIOD;
  }
  if (param.delta < 0.) {
    return LBFGSERR_INVALID_DELTA;
  }
  if (param.min_step < 0.) {
    return LBFGSERR_INVALID_MINSTEP;
  }
  if (param.max_step < param.min_step) {
    return LBFGSERR_INVALID_MAXSTEP;
  }
  if (param.ftol < 0.) {
    return LBFGSERR_INVALID_FTOL;
  }
  if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE ||
      param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
    if (param.wolfe <= param.ftol || 1. <= param.wolfe) {
      return LBFGSERR_INVALID_WOLFE;
    }
  }
  if (param.gtol < 0.) {
    return LBFGSERR_INVALID_GTOL;
  }
  if (param.xtol < 0.) {
    return LBFGSERR_INVALID_XTOL;
  }
  if (param.max_linesearch <= 0) {
    return LBFGSERR_INVALID_MAXLINESEARCH;
  }
  return 0;
}

int lbfgs(
          int n,
          lbfgsfloatval_t *x,
          lbfgsfloatval_t *ptr_fx,
          lbfgs_evaluate_t proc_evaluate,
          lbfgs_progress_t proc_progress,
          void *instance,
          lbfgs_parameter_t *_param
          )
{
  int ret;
  int i, j, k, ls, end, bound;
  lbfgsfloatval_t step;

  /* Constant parameters and their default values. */
  lbfgs_parameter_t param = (_param != NULL) ? (*_param) : _def_param;
  const int m = param.m;

  lbfgsfloatval_t *xp = NULL;
  lbfgsfloatval_t *g = NULL, *gp = NULL, *pg = NULL;
  lbfgsfloatval_t *d = NULL, *w = NULL, *pf = NULL;
  iteration_data_t *lm = NULL, *it = NULL;
  lbfgsfloatval_t ys, yy;
  lbfgsfloatval_t xnorm, gnorm, beta;
  lbfgsfloatval_t fx = 0.;
  lbfgsfloatval_t rate = 0.;
  line_search_proc linesearch = line_search_morethuente;

  /* Construct a callback data. */
  callback_data_t cd;
  cd.n = n;
  cd.instance = instance;
  cd.proc_evaluate = proc_evaluate;
  cd.proc_progress = proc_progress;

  /* check common params */
  ret = check_params(n,param);
  if (ret < 0) {
    return ret;
  }

  /* check params specific to lbfgs() */
  if (param.orthantwise_c < 0.) {
    return LBFGSERR_INVALID_ORTHANTWISE;
  }
  if (param.orthantwise_start < 0 || n < param.orthantwise_start) {
    return LBFGSERR_INVALID_ORTHANTWISE_START;
  }
  if (param.orthantwise_end < 0) {
    param.orthantwise_end = n;
  }
  if (n < param.orthantwise_end) {
    return LBFGSERR_INVALID_ORTHANTWISE_END;
  }
  if (param.orthantwise_c != 0.) {
    switch (param.linesearch) {
    case LBFGS_LINESEARCH_BACKTRACKING:
      linesearch = line_search_backtracking_owlqn;
      break;
    default:
      /* Only the backtracking method is available. */
      return LBFGSERR_INVALID_LINESEARCH;
    }
  } else {
    switch (param.linesearch) {
    case LBFGS_LINESEARCH_MORETHUENTE:
      linesearch = line_search_morethuente;
      break;
    case LBFGS_LINESEARCH_BACKTRACKING_ARMIJO:
    case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
    case LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE:
      linesearch = line_search_backtracking;
      break;
    default:
      return LBFGSERR_INVALID_LINESEARCH;
    }
  }

  /* Allocate working space. */
  xp = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  g = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  gp = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  d = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  w = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  if (xp == NULL || g == NULL || gp == NULL || d == NULL || w == NULL) {
    ret = LBFGSERR_OUTOFMEMORY;
    goto lbfgs_exit;
  }

  if (param.orthantwise_c != 0.) {
    /* Allocate working space for OW-LQN. */
    pg = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
    if (pg == NULL) {
      ret = LBFGSERR_OUTOFMEMORY;
      goto lbfgs_exit;
    }
  }

  /* Allocate limited memory storage. */
  lm = (iteration_data_t*)vecalloc(m * sizeof(iteration_data_t));
  if (lm == NULL) {
    ret = LBFGSERR_OUTOFMEMORY;
    goto lbfgs_exit;
  }

  /* Initialize the limited memory. */
  for (i = 0;i < m;++i) {
    it = &lm[i];
    it->alpha = 0;
    it->ys = 0;
    it->s = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
    it->y = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
    if (it->s == NULL || it->y == NULL) {
      ret = LBFGSERR_OUTOFMEMORY;
      goto lbfgs_exit;
    }
  }

  /* Allocate an array for storing previous values of the objective function. */
  if (0 < param.past) {
    pf = (lbfgsfloatval_t*)vecalloc(param.past * sizeof(lbfgsfloatval_t));
  }

  /* Evaluate the function value and its gradient. */
  fx = cd.proc_evaluate(cd.instance, x, g, cd.n, 0);
  if (verbose > 2){
    printf("<lbfgs()>\n");
    print_fxxdx(fx,x,g,cd.n);
  }
  if (0. != param.orthantwise_c) {
    /* Compute the L1 norm of the variable and add it to the object value. */
    xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
    fx += xnorm * param.orthantwise_c;
    owlqn_pseudo_gradient(
                          pg, x, g, n,
                          param.orthantwise_c,
                          param.orthantwise_start, param.orthantwise_end
                          );
  }

  /* Store the initial value of the objective function. */
  if (pf != NULL) {
    pf[0] = fx;
  }

  /*
    Compute the direction;
    we assume the initial hessian matrix H_0 as the identity matrix.
  */
  if (param.orthantwise_c == 0.) {
    vecncpy(d, g, n);
  } else {
    vecncpy(d, pg, n);
  }

  /*
    Make sure that the initial variables are not a minimizer.
  */
  vec2norm(&xnorm, x, n);
  if (param.orthantwise_c == 0.) {
    vec2norm(&gnorm, g, n);
  } else {
    vec2norm(&gnorm, pg, n);
  }
  if (xnorm < 1.0) xnorm = 1.0;
  if (gnorm / xnorm <= param.epsilon) {
    ret = LBFGS_ALREADY_MINIMIZED;
    goto lbfgs_exit;
  }

  /* Compute the initial step:
     step = 1.0 / sqrt(vecdot(d, d, n))
  */
  vec2norminv(&step, d, n);

  k = 1;
  end = 0;
  for (;;) {
    /* Store the current position and gradient vectors. */
    veccpy(xp, x, n);
    veccpy(gp, g, n);

    /* Search for an optimal step. */
    if (param.orthantwise_c == 0.) {
      ls = linesearch(n, x, &fx, g, d, &step, xp, gp, w, &cd, &param);
    } else {
      ls = linesearch(n, x, &fx, g, d, &step, xp, pg, w, &cd, &param);
      owlqn_pseudo_gradient(
                            pg, x, g, n,
                            param.orthantwise_c,
                            param.orthantwise_start, param.orthantwise_end
                            );
    }
    if (ls < 0) {
      /* Revert to the previous point. */
      veccpy(x, xp, n);
      veccpy(g, gp, n);
      ret = ls;
      if (verbose > 1){
        printf("<linesearch> Stopping b/c :\n");
        print_lbfgs_error(ls);
      }
      goto lbfgs_exit;
    }

    /* Compute x and g norms. */
    vec2norm(&xnorm, x, n);
    if (param.orthantwise_c == 0.) {
      vec2norm(&gnorm, g, n);
    } else {
      vec2norm(&gnorm, pg, n);
    }

    /* Report the progress. */
    if (cd.proc_progress) {
      if ((ret = cd.proc_progress(cd.instance, x, g, fx, xnorm, gnorm, step, cd.n, k, ls))) {
        if (verbose > 1){
          printf("<lbfgs()> Stopping b/c cd.proc_progress (%d)\n", ret);
        }
        goto lbfgs_exit;
      }
    }

    /* Count number of function evaluations */
    if ((param.max_evaluations != 0)&&(nEvaluation > param.max_evaluations)) {
      if (verbose > 1){
        printf("<lbfgs()> Stopping b/c exceeded max number of function evaluations\n");
      }
      ret = LBFGSERR_MAXIMUMEVALUATION;
      goto lbfgs_exit;
    }
    /*
      Convergence test.
      The criterion is given by the following formula:
      |g(x)| / \max(1, |x|) < \epsilon
    */
    if (xnorm < 1.0) xnorm = 1.0;
    if (gnorm / xnorm <= param.epsilon) {
      if (verbose > 1){
        printf("<lbfgs()> Stopping b/c gnorm(%f)/xnorm(%f) <= param.epsilon (%f)\n",
               gnorm, xnorm, param.epsilon);
      }
      /* Convergence. */
      ret = LBFGS_SUCCESS;
      break;
    }

    /*
      Test for stopping criterion.
      The criterion is given by the following formula:
      (f(past_x) - f(x)) / f(x) < \delta
    */
    if (pf != NULL) {
      /* We don't test the stopping criterion while k < past. */
      if (param.past <= k) {
        /* Compute the relative improvement from the past. */
        rate = (pf[k % param.past] - fx) / fx;

        /* The stopping criterion. */
        if (rate < param.delta) {
          if (verbose > 1){
            printf("<lbfgs()> Stopping b/c rate (%f) < param.delta (%f)\n",
                   rate, param.delta);
          }
          ret = LBFGS_STOP;
          break;
        }
      }

      /* Store the current value of the objective function. */
      pf[k % param.past] = fx;
    }

    if (param.max_iterations != 0 && param.max_iterations < k+1) {
      if (verbose > 1){
        printf("<lbfgs()> Stopping b/c param.max_iterations (%d) < k+1 (%d)\n",
               param.max_iterations, k+1);
      }
      /* Maximum number of iterations. */
      ret = LBFGSERR_MAXIMUMITERATION;
      break;
    }

    /*
      Update vectors s and y:
      s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
      y_{k+1} = g_{k+1} - g_{k}.
    */
    it = &lm[end];
    vecdiff(it->s, x, xp, n);
    vecdiff(it->y, g, gp, n);

    /*
      Compute scalars ys and yy:
      ys = y^t \cdot s = 1 / \rho.
      yy = y^t \cdot y.
      Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
    */
    vecdot(&ys, it->y, it->s, n);
    vecdot(&yy, it->y, it->y, n);
    it->ys = ys;

    /*
      Recursive formula to compute dir = -(H \cdot g).
      This is described in page 779 of:
      Jorge Nocedal.
      Updating Quasi-Newton Matrices with Limited Storage.
      Mathematics of Computation, Vol. 35, No. 151,
      pp. 773--782, 1980.
    */
    bound = (m <= k) ? m : k;
    ++k;
    end = (end + 1) % m;

    /* Compute the steepest direction. */
    if (param.orthantwise_c == 0.) {
      /* Compute the negative of gradients. */
      vecncpy(d, g, n);
    } else {
      vecncpy(d, pg, n);
    }

    j = end;
    for (i = 0;i < bound;++i) {
      j = (j + m - 1) % m;    /* if (--j == -1) j = m-1; */
      it = &lm[j];
      /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
      vecdot(&it->alpha, it->s, d, n);
      it->alpha /= it->ys;
      /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
      vecadd(d, it->y, -it->alpha, n);
    }

    vecscale(d, ys / yy, n);

    for (i = 0;i < bound;++i) {
      it = &lm[j];
      /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}. */
      vecdot(&beta, it->y, d, n);
      beta /= it->ys;
      /* \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
      vecadd(d, it->s, it->alpha - beta, n);
      j = (j + 1) % m;        /* if (++j == m) j = 0; */
    }

    /*
      Constrain the search direction for orthant-wise updates.
    */
    if (param.orthantwise_c != 0.) {
      for (i = param.orthantwise_start;i < param.orthantwise_end;++i) {
        if (d[i] * pg[i] >= 0) {
          d[i] = 0;
        }
      }
    }

    /*
      Now the search direction d is ready. We try step = 1 first.
    */
    step = 1.0;
  }

 lbfgs_exit:
  /* Return the final value of the objective function. */
  if (ptr_fx != NULL) {
    *ptr_fx = fx;
  }

  if(verbose){
    printf("<lbfgs()>\n");
    print_lbfgs_error(ret);
  }
  vecfree(pf);

  /* Free memory blocks used by this function. */
  if (lm != NULL) {
    for (i = 0;i < m;++i) {
      vecfree(lm[i].s);
      vecfree(lm[i].y);
    }
    vecfree(lm);
  }
  vecfree(pg);
  vecfree(w);
  vecfree(d);
  vecfree(gp);
  vecfree(g);
  vecfree(xp);

  return ret;
}


int cg(
       int n,
       lbfgsfloatval_t *x,
       lbfgsfloatval_t *ptr_fx,
       lbfgs_evaluate_t proc_evaluate,
       lbfgs_progress_t proc_progress,
       void *instance,
       lbfgs_parameter_t *_param
       )
{
  int ret;
  int i, j, k, ls, end, bound;
  lbfgsfloatval_t step;

  /* Constant parameters and their default values. */
  lbfgs_parameter_t param = (_param != NULL) ? (*_param) : _def_param;

  lbfgsfloatval_t *xp = NULL;
  lbfgsfloatval_t *g = NULL, *gp = NULL, *pg = NULL;
  lbfgsfloatval_t *d = NULL, *dp = NULL, *w = NULL, *pf = NULL;
  lbfgsfloatval_t *tmp = NULL;
  lbfgsfloatval_t xnorm, gnorm;
  lbfgsfloatval_t B, gptgp, gtg, gtgp, gnum, gden, B_FR, B_PR;
  lbfgsfloatval_t fx = 0.;
  lbfgsfloatval_t rate = 0.;
  line_search_proc linesearch = line_search_morethuente;

  /* Construct a callback data. */
  callback_data_t cd;
  cd.n = n;
  cd.instance = instance;
  cd.proc_evaluate = proc_evaluate;
  cd.proc_progress = proc_progress;

  /* check common params */
  ret = check_params(n,param);
  if (ret < 0) {
    return ret;
  }
  /* check CG specific params */
  if (param.momentum < 0 || param.momentum > 3 ){
    return CGERR_INVALID_MOMENTUM;
  }
  switch (param.linesearch) {
  case LBFGS_LINESEARCH_MORETHUENTE:
    linesearch = line_search_morethuente;
    break;
  case LBFGS_LINESEARCH_BACKTRACKING_ARMIJO:
  case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
  case LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE:
    linesearch = line_search_backtracking;
    break;
  default:
    return LBFGSERR_INVALID_LINESEARCH;
  }


  /* Allocate working space. */
  xp = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  g  = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  gp = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  d  = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  dp = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  w  = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  tmp  = (lbfgsfloatval_t*)vecalloc(n * sizeof(lbfgsfloatval_t));
  if (xp == NULL || g == NULL || gp == NULL ||
      d == NULL || dp == NULL || w == NULL || tmp == NULL) {
    ret = LBFGSERR_OUTOFMEMORY;
    goto cg_exit;
  }

  /* Allocate an array for storing previous values of the objective function. */
  if (0 < param.past) {
    pf = (lbfgsfloatval_t*)vecalloc(param.past * sizeof(lbfgsfloatval_t));
  }

  /* Evaluate the function value and its gradient. */
  fx = cd.proc_evaluate(cd.instance, x, g, cd.n, 0);
  if (verbose > 2){
    printf("<cg()>\n");
    print_fxxdx(fx,x,g,cd.n);
  }
  /* used to compute the momentum  term for CG */
  vecdot(&gtg,g,g,n);

  /* Store the initial value of the objective function. */
  if (pf != NULL) {
    pf[0] = fx;
  }

  /*
    Compute the inital search direction (the negative gradient)
  */
  vecncpy(d, g, n);

  /*
    Make sure that the initial variables are not a minimizer.
  */
  vec2norm(&xnorm, x, n);
  vec2norm(&gnorm, g, n);


  if (xnorm < 1.0) xnorm = 1.0;
  if (gnorm / xnorm <= param.epsilon) {
    ret = LBFGS_ALREADY_MINIMIZED;
    goto cg_exit;
  }

  /* Compute the initial step:
     1 / |d| + 1
     from minfunc: 
     t = min(1,1/sum(abs(g)));
  */
  vec1norminv(&step, d, n);
  step = min2(1,step);

  k = 1;
  end = 0;
  for (;;) {
    /* Store the current position and gradient vectors. */
    veccpy(xp, x, n);
    veccpy(gp, g, n);
    veccpy(dp, d, n);

    /* Search for an optimal step. */
    ls = linesearch(n, x, &fx, g, d, &step, xp, gp, w, &cd, &param);

    if (ls < 0) {
      /* Revert to the previous point. */
      veccpy(x, xp, n);
      veccpy(g, gp, n);
      ret = ls;
      if (verbose > 1){
        printf("<linesearch()> Stopping b/c :\n");
        print_lbfgs_error(ls);
      }
      // goto cg_exit;
    }

    /* Compute x and g norms. */
    vec2norm(&xnorm, x, n);
    vec2norm(&gnorm, g, n);

    /* Report the progress. */
    if (cd.proc_progress) {
      if ((ret = cd.proc_progress(cd.instance, x, g, fx, xnorm, gnorm, step, cd.n, k, ls))) {
        if (verbose > 1){
          printf("<cg()> Stopping b/c cd.proc_progress (%d)\n", ret);
        }
        goto cg_exit;
      }
    }

    /* Count number of function evaluations */
    if ((param.max_evaluations != 0)&&(nEvaluation > param.max_evaluations)) {
      if (verbose > 1){
        printf("<cg()> Stopping b/c exceeded max number of function evaluations\n");
      }
      goto cg_exit;
    }
    /*
      Convergence test.
      The criterion is given by the following formula:
      |g(x)| / \max(1, |x|) < \epsilon
    */
    if (xnorm < 1.0) xnorm = 1.0;
    if (gnorm / xnorm <= param.epsilon) {
      if (verbose > 1){
        printf("<cg()> Stopping b/c gnorm(%f)/xnorm(%f) <= param.epsilon (%f)\n",
               gnorm, xnorm, param.epsilon);
      }
      /* Convergence. */
      ret = LBFGS_SUCCESS;
      break;
    }

    /*
      Test for stopping criterion.
      The criterion is given by the following formula:
      (f(past_x) - f(x)) / f(x) < \delta
    */
    if (pf != NULL) {
      /* We don't test the stopping criterion while k < past. */
      if (param.past <= k) {
        /* Compute the relative improvement from the past. */
        rate = (pf[k % param.past] - fx) / fx;

        /* The stopping criterion. */
        if (rate < param.delta) {
          if (verbose > 1){
            printf("<cg()> Stopping b/c rate (%f) < param.delta (%f)\n",
                   rate, param.delta);
          }
          ret = LBFGS_STOP;
          break;
        }
      }

      /* Store the current value of the objective function. */
      pf[k % param.past] = fx;
    }

    if (param.max_iterations != 0 && param.max_iterations < k+1) {
      if (verbose > 1){
        printf("<cg()> Stopping b/c param.max_iterations (%d) < k+1 (%d)\n",
               param.max_iterations, k+1);
      }
      /* Maximum number of iterations. */
      ret = LBFGSERR_MAXIMUMITERATION;
      break;
    }

    if (k > 1)
    {
      /* compute 'momentum' term (following min func) */
      if (param.momentum != CG_HESTENES_STIEFEL) {
	vecdot(&gtg, g, g, n);
      }
      switch(param.momentum) {
      case CG_FLETCHER_REEVES :
	/* B = (g'*g)/(gp'*gp) */
	B = gtg / gptgp;
	break;
      case CG_POLAK_RIBIERE :
	/* B = (g'*(g-gp)) /(gp'*gp);*/
	vecdiff(tmp,g,gp,n);
	vecdot(&gnum,g,tmp,n);
	B = gnum / gptgp;
	break;
      case CG_HESTENES_STIEFEL :
	/* B = (g'*(g-gp))/((g-gp)'*d); */
	vecdiff(tmp,g,gp,n);
	vecdot(&gnum,g,tmp,n);
	vecdot(&gden,tmp,d,n);
	B = gnum / gden;
	break;
      case CG_GILBERT_NOCEDAL :
	/* B_FR = (g'*(g-gp)) /(gp'*gp); */
	/* B_PR = (g'*g-(g'*gp))/(gp'*gp); */
	/* B = max(-B_FR,min(B_PR,B_FR)); */
	vecdiff(tmp,g,gp,n);   /*  g-gp */
	vecdot(&gnum,g,tmp,n); /*  g'*(g-gp) */
	B_FR = gnum / gptgp;   /* (g'*(g-gp)) /(gp'*gp) */
	vecdot(&gtgp,g,gp,n);   /*  g'*gp */
	gnum = gtg - gtgp;     /*  g'*g-(g'*gp) */
	B_PR = gnum / gptgp;   /* (g'*g-(g'*gp))/(gp'*gp) */
	B = max2(-B_FR,min2(B_PR,B_FR));
	break;
      default :
	ret = CGERR_INVALID_MOMENTUM;
	break;
      }

      /* Compute the steepest direction. */
      /* Compute the negative of gradients. */
      vecncpy(d, g, n);
      
      /* add the 'momentum' term */
      /* d_1 = -g_1 + B*d_0 */
      vecadd(d, dp, B, n);
    }
    if (param.momentum != CG_HESTENES_STIEFEL) {
      /* store val for next iteration */
      gptgp = gtg;
    }

    /* increment the number of iterations */
    ++k;

    /*
      Now the search direction d is ready. We try step = 1 first.
    */
    step = 1.0;
  }

 cg_exit:
  /* Return the final value of the objective function. */
  if (ptr_fx != NULL) {
    *ptr_fx = fx;
  }

  if(verbose){
    print_lbfgs_error(ret);
  }

  vecfree(pf);
  vecfree(pg);
  vecfree(w);
  vecfree(d);
  vecfree(gp);
  vecfree(g);
  vecfree(xp);
  vecfree(dp);
  vecfree(tmp);

  return ret;
}

static int line_search_backtracking(
                                    int n,
                                    lbfgsfloatval_t *x,
                                    lbfgsfloatval_t *f,
                                    lbfgsfloatval_t *g,
                                    lbfgsfloatval_t *s,
                                    lbfgsfloatval_t *stp,
                                    const lbfgsfloatval_t* xp,
                                    const lbfgsfloatval_t* gp,
                                    lbfgsfloatval_t *wp,
                                    callback_data_t *cd,
                                    const lbfgs_parameter_t *param
                                    )
{
  int count = 0;
  lbfgsfloatval_t width, dg;
  lbfgsfloatval_t finit, dginit = 0., dgtest;
  const lbfgsfloatval_t dec = 0.5, inc = 2.1;

  /* Check the input parameters for errors. */
  if (*stp <= 0.) {
    return LBFGSERR_INVALIDPARAMETERS;
  }

  /* Compute the initial gradient in the search direction. */
  vecdot(&dginit, g, s, n);

  /* Make sure that s points to a descent direction. */
  if (0 < dginit) {
    return LBFGSERR_INCREASEGRADIENT;
  }

  /* The initial value of the objective function. */
  finit = *f;
  dgtest = param->ftol * dginit;

  for (;;) {
    veccpy(x, xp, n);
    vecadd(x, s, *stp, n);

    /* Evaluate the function and gradient values. */
    *f = cd->proc_evaluate(cd->instance, x, g, cd->n, *stp);
    
    if (verbose > 2){
      printf("<line_search_backtracking()>\n");
      print_linesearch_type(param->linesearch);
      print_fxxdx(*f,x,g,cd->n);
    }

    ++count;

    if (*f > finit + *stp * dgtest) {
      width = dec;
    } else {
      /* The sufficient decrease condition (Armijo condition). */
      if (param->linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO) {
        /* Exit with the Armijo condition. */
        return count;
      }

      /* Check the Wolfe condition. */
      vecdot(&dg, g, s, n);
      if (dg < param->wolfe * dginit) {
        width = inc;
      } else {
        if(param->linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE) {
          /* Exit with the regular Wolfe condition. */
          return count;
        }

        /* Check the strong Wolfe condition. */
        if(dg > -param->wolfe * dginit) {
          width = dec;
        } else {
          /* Exit with the strong Wolfe condition. */
          return count;
        }
      }
    }

    if (*stp < param->min_step) {
      /* The step is the minimum value. */
      return LBFGSERR_MINIMUMSTEP;
    }
    if (*stp > param->max_step) {
      /* The step is the maximum value. */
      return LBFGSERR_MAXIMUMSTEP;
    }
    if (param->max_linesearch <= count) {
      /* Maximum number of iteration. */
      return LBFGSERR_MAXIMUMLINESEARCH;
    }

    (*stp) *= width;
  }
}



static int line_search_backtracking_owlqn(
                                          int n,
                                          lbfgsfloatval_t *x,
                                          lbfgsfloatval_t *f,
                                          lbfgsfloatval_t *g,
                                          lbfgsfloatval_t *s,
                                          lbfgsfloatval_t *stp,
                                          const lbfgsfloatval_t* xp,
                                          const lbfgsfloatval_t* gp,
                                          lbfgsfloatval_t *wp,
                                          callback_data_t *cd,
                                          const lbfgs_parameter_t *param
                                          )
{
  int i, count = 0;
  lbfgsfloatval_t width = 0.5, norm = 0.;
  lbfgsfloatval_t finit = *f, dgtest;

  /* Check the input parameters for errors. */
  if (*stp <= 0.) {
    return LBFGSERR_INVALIDPARAMETERS;
  }

  /* Choose the orthant for the new point. */
  for (i = 0;i < n;++i) {
    wp[i] = (xp[i] == 0.) ? -gp[i] : xp[i];
  }

  for (;;) {
    /* Update the current point. */
    veccpy(x, xp, n);
    vecadd(x, s, *stp, n);

    /* The current point is projected onto the orthant. */
    owlqn_project(x, wp, param->orthantwise_start, param->orthantwise_end);

    /* Evaluate the function and gradient values. */
    *f = cd->proc_evaluate(cd->instance, x, g, cd->n, *stp);

    if (verbose > 2){
      printf("<line_search_backtracking_owlqn()>\n");
      print_linesearch_type(param->linesearch);
      print_fxxdx(*f,x,g,cd->n);
    }
    
    /* Compute the L1 norm of the variables and add it to the object value. */
    norm = owlqn_x1norm(x, param->orthantwise_start, param->orthantwise_end);
    *f += norm * param->orthantwise_c;

    ++count;

    dgtest = 0.;
    for (i = 0;i < n;++i) {
      dgtest += (x[i] - xp[i]) * gp[i];
    }

    if (*f <= finit + param->ftol * dgtest) {
      /* The sufficient decrease condition. */
      return count;
    }

    if (*stp < param->min_step) {
      /* The step is the minimum value. */
      return LBFGSERR_MINIMUMSTEP;
    }
    if (*stp > param->max_step) {
      /* The step is the maximum value. */
      return LBFGSERR_MAXIMUMSTEP;
    }
    if (param->max_linesearch <= count) {
      /* Maximum number of iteration. */
      return LBFGSERR_MAXIMUMLINESEARCH;
    }

    (*stp) *= width;
  }
}



static int line_search_morethuente(
                                   int n,
                                   lbfgsfloatval_t *x,
                                   lbfgsfloatval_t *f,
                                   lbfgsfloatval_t *g,
                                   lbfgsfloatval_t *s,
                                   lbfgsfloatval_t *stp,
                                   const lbfgsfloatval_t* xp,
                                   const lbfgsfloatval_t* gp,
                                   lbfgsfloatval_t *wa,
                                   callback_data_t *cd,
                                   const lbfgs_parameter_t *param
                                   )
{
  int count = 0;
  int brackt, stage1, uinfo = 0;
  lbfgsfloatval_t dg;
  lbfgsfloatval_t stx, fx, dgx;
  lbfgsfloatval_t sty, fy, dgy;
  lbfgsfloatval_t fxm, dgxm, fym, dgym, fm, dgm;
  lbfgsfloatval_t finit, ftest1, dginit, dgtest;
  lbfgsfloatval_t width, prev_width;
  lbfgsfloatval_t stmin, stmax;

  /* Check the input parameters for errors. */
  if (*stp <= 0.) {
    return LBFGSERR_INVALIDPARAMETERS;
  }

  /* Compute the initial gradient in the search direction. */
  vecdot(&dginit, g, s, n);

  /* Make sure that s points to a descent direction. */
  if (0 < dginit) {
    return LBFGSERR_INCREASEGRADIENT;
  }

  /* Initialize local variables. */
  brackt = 0;
  stage1 = 1;
  finit = *f;
  dgtest = param->ftol * dginit;
  width = param->max_step - param->min_step;
  prev_width = 2.0 * width;

  /*
    The variables stx, fx, dgx contain the values of the step,
    function, and directional derivative at the best step.
    The variables sty, fy, dgy contain the value of the step,
    function, and derivative at the other endpoint of
    the interval of uncertainty.
    The variables stp, f, dg contain the values of the step,
    function, and derivative at the current step.
  */
  stx = sty = 0.;
  fx = fy = finit;
  dgx = dgy = dginit;

  for (;;) {
    /*
      Set the minimum and maximum steps to correspond to the
      present interval of uncertainty.
    */
    if (brackt) {
      stmin = min2(stx, sty);
      stmax = max2(stx, sty);
    } else {
      stmin = stx;
      stmax = *stp + 4.0 * (*stp - stx);
    }

    /* Clip the step in the range of [stpmin, stpmax]. */
    if (*stp < param->min_step) *stp = param->min_step;
    if (param->max_step < *stp) *stp = param->max_step;

    /*
      If an unusual termination is to occur then let
      stp be the lowest point obtained so far.
    */
    if ((brackt && ((*stp <= stmin || stmax <= *stp) || param->max_linesearch <= count + 1 || uinfo != 0)) || (brackt && (stmax - stmin <= param->xtol * stmax))) {
      *stp = stx;
    }

    /*
      Compute the current value of x:
      x <- x + (*stp) * s.
    */
    veccpy(x, xp, n);
    vecadd(x, s, *stp, n);

    /* Evaluate the function and gradient values. */
    *f = cd->proc_evaluate(cd->instance, x, g, cd->n, *stp);

    if (verbose > 2){
      printf("<line_search_morethuente()>\n");
      print_linesearch_type(param->linesearch);
      print_fxxdx(*f,x,g,cd->n);
    }

    vecdot(&dg, g, s, n);

    ftest1 = finit + *stp * dgtest;
    ++count;

    /* Test for errors and convergence. */
    if (brackt && ((*stp <= stmin || stmax <= *stp) || uinfo != 0)) {
      /* Rounding errors prevent further progress. */
      return LBFGSERR_ROUNDING_ERROR;
    }
    if (*stp == param->max_step && *f <= ftest1 && dg <= dgtest) {
      /* The step is the maximum value. */
      return LBFGSERR_MAXIMUMSTEP;
    }
    if (*stp == param->min_step && (ftest1 < *f || dgtest <= dg)) {
      /* The step is the minimum value. */
      return LBFGSERR_MINIMUMSTEP;
    }
    if (brackt && (stmax - stmin) <= param->xtol * stmax) {
      /* Relative width of the interval of uncertainty is at most xtol. */
      return LBFGSERR_WIDTHTOOSMALL;
    }
    if (param->max_linesearch <= count) {
      /* Maximum number of iteration. */
      return LBFGSERR_MAXIMUMLINESEARCH;
    }
    if (*f <= ftest1 && fabs(dg) <= param->gtol * (-dginit)) {
      /* The sufficient decrease condition and the directional derivative condition hold. */
      return count;
    }

    /*
      In the first stage we seek a step for which the modified
      function has a nonpositive value and nonnegative derivative.
    */
    if (stage1 && *f <= ftest1 && min2(param->ftol, param->gtol) * dginit <= dg) {
      stage1 = 0;
    }

    /*
      A modified function is used to predict the step only if
      we have not obtained a step for which the modified
      function has a nonpositive function value and nonnegative
      derivative, and if a lower function value has been
      obtained but the decrease is not sufficient.
    */
    if (stage1 && ftest1 < *f && *f <= fx) {
      /* Define the modified function and derivative values. */
      fm = *f - *stp * dgtest;
      fxm = fx - stx * dgtest;
      fym = fy - sty * dgtest;
      dgm = dg - dgtest;
      dgxm = dgx - dgtest;
      dgym = dgy - dgtest;

      /*
        Call update_trial_interval() to update the interval of
        uncertainty and to compute the new step.
      */
      uinfo = update_trial_interval(
                                    &stx, &fxm, &dgxm,
                                    &sty, &fym, &dgym,
                                    stp, &fm, &dgm,
                                    stmin, stmax, &brackt
                                    );

      /* Reset the function and gradient values for f. */
      fx = fxm + stx * dgtest;
      fy = fym + sty * dgtest;
      dgx = dgxm + dgtest;
      dgy = dgym + dgtest;
    } else {
      /*
        Call update_trial_interval() to update the interval of
        uncertainty and to compute the new step.
      */
      uinfo = update_trial_interval(
                                    &stx, &fx, &dgx,
                                    &sty, &fy, &dgy,
                                    stp, f, &dg,
                                    stmin, stmax, &brackt
                                    );
    }

    /*
      Force a sufficient decrease in the interval of uncertainty.
    */
    if (brackt) {
      if (0.66 * prev_width <= fabs(sty - stx)) {
        *stp = stx + 0.5 * (sty - stx);
      }
      prev_width = width;
      width = fabs(sty - stx);
    }
  }

  return LBFGSERR_LOGICERROR;
}



/**
 * Define the local variables for computing minimizers.
 */
#define USES_MINIMIZER                                  \
  lbfgsfloatval_t a, d, gamma, theta, p, q, r, s;

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param  cm      The minimizer of the interpolated cubic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 *  @param  du      The value of f'(v).
 */
#define CUBIC_MINIMIZER(cm, u, fu, du, v, fv, dv)       \
  d = (v) - (u);                                        \
  theta = ((fu) - (fv)) * 3 / d + (du) + (dv);          \
  p = fabs(theta);                                      \
  q = fabs(du);                                         \
  r = fabs(dv);                                         \
  s = max3(p, q, r);                                    \
  /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */  \
  a = theta / s;                                        \
  gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s));    \
  if ((v) < (u)) gamma = -gamma;                        \
  p = gamma - (du) + theta;                             \
  q = gamma - (du) + gamma + (dv);                      \
  r = p / q;                                            \
  (cm) = (u) + r * d;

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param  cm      The minimizer of the interpolated cubic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 *  @param  du      The value of f'(v).
 *  @param  xmin    The maximum value.
 *  @param  xmin    The minimum value.
 */
#define CUBIC_MINIMIZER2(cm, u, fu, du, v, fv, dv, xmin, xmax)  \
  d = (v) - (u);                                                \
  theta = ((fu) - (fv)) * 3 / d + (du) + (dv);                  \
  p = fabs(theta);                                              \
  q = fabs(du);                                                 \
  r = fabs(dv);                                                 \
  s = max3(p, q, r);                                            \
  /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */          \
  a = theta / s;                                                \
  gamma = s * sqrt(max2(0, a * a - ((du) / s) * ((dv) / s)));   \
  if ((u) < (v)) gamma = -gamma;                                \
  p = gamma - (dv) + theta;                                     \
  q = gamma - (dv) + gamma + (du);                              \
  r = p / q;                                                    \
  if (r < 0. && gamma != 0.) {                                  \
    (cm) = (v) - r * d;                                         \
  } else if (a < 0) {                                           \
    (cm) = (xmax);                                              \
  } else {                                                      \
    (cm) = (xmin);                                              \
  }

/**
 * Find a minimizer of an interpolated quadratic function.
 *  @param  qm      The minimizer of the interpolated quadratic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 */
#define QUARD_MINIMIZER(qm, u, fu, du, v, fv)                   \
  a = (v) - (u);                                                \
  (qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;

/**
 * Find a minimizer of an interpolated quadratic function.
 *  @param  qm      The minimizer of the interpolated quadratic.
 *  @param  u       The value of one point, u.
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  dv      The value of f'(v).
 */
#define QUARD_MINIMIZER2(qm, u, du, v, dv)      \
  a = (u) - (v);                                \
  (qm) = (v) + (dv) / ((dv) - (du)) * a;

/**
 * Update a safeguarded trial value and interval for line search.
 *
 *  The parameter x represents the step with the least function value.
 *  The parameter t represents the current step. This function assumes
 *  that the derivative at the point of x in the direction of the step.
 *  If the bracket is set to true, the minimizer has been bracketed in
 *  an interval of uncertainty with endpoints between x and y.
 *
 *  @param  x       The pointer to the value of one endpoint.
 *  @param  fx      The pointer to the value of f(x).
 *  @param  dx      The pointer to the value of f'(x).
 *  @param  y       The pointer to the value of another endpoint.
 *  @param  fy      The pointer to the value of f(y).
 *  @param  dy      The pointer to the value of f'(y).
 *  @param  t       The pointer to the value of the trial value, t.
 *  @param  ft      The pointer to the value of f(t).
 *  @param  dt      The pointer to the value of f'(t).
 *  @param  tmin    The minimum value for the trial value, t.
 *  @param  tmax    The maximum value for the trial value, t.
 *  @param  brackt  The pointer to the predicate if the trial value is
 *                  bracketed.
 *  @retval int     Status value. Zero indicates a normal termination.
 *
 *  @see
 *      Jorge J. More and David J. Thuente. Line search algorithm with
 *      guaranteed sufficient decrease. ACM Transactions on Mathematical
 *      Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
 */
static int update_trial_interval(
                                 lbfgsfloatval_t *x,
                                 lbfgsfloatval_t *fx,
                                 lbfgsfloatval_t *dx,
                                 lbfgsfloatval_t *y,
                                 lbfgsfloatval_t *fy,
                                 lbfgsfloatval_t *dy,
                                 lbfgsfloatval_t *t,
                                 lbfgsfloatval_t *ft,
                                 lbfgsfloatval_t *dt,
                                 const lbfgsfloatval_t tmin,
                                 const lbfgsfloatval_t tmax,
                                 int *brackt
                                 )
{
  int bound;
  int dsign = fsigndiff(dt, dx);
  lbfgsfloatval_t mc; /* minimizer of an interpolated cubic. */
  lbfgsfloatval_t mq; /* minimizer of an interpolated quadratic. */
  lbfgsfloatval_t newt;   /* new trial value. */
  USES_MINIMIZER;     /* for CUBIC_MINIMIZER and QUARD_MINIMIZER. */

  /* Check the input parameters for errors. */
  if (*brackt) {
    if (*t <= min2(*x, *y) || max2(*x, *y) <= *t) {
      /* The trival value t is out of the interval. */
      return LBFGSERR_OUTOFINTERVAL;
    }
    if (0. <= *dx * (*t - *x)) {
      /* The function must decrease from x. */
      return LBFGSERR_INCREASEGRADIENT;
    }
    if (tmax < tmin) {
      /* Incorrect tmin and tmax specified. */
      return LBFGSERR_INCORRECT_TMINMAX;
    }
  }

  /*
    Trial value selection.
  */
  if (*fx < *ft) {
    /*
      Case 1: a higher function value.
      The minimum is brackt. If the cubic minimizer is closer
      to x than the quadratic one, the cubic one is taken, else
      the average of the minimizers is taken.
    */
    *brackt = 1;
    bound = 1;
    CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
    QUARD_MINIMIZER(mq, *x, *fx, *dx, *t, *ft);
    if (fabs(mc - *x) < fabs(mq - *x)) {
      newt = mc;
    } else {
      newt = mc + 0.5 * (mq - mc);
    }
  } else if (dsign) {
    /*
      Case 2: a lower function value and derivatives of
      opposite sign. The minimum is brackt. If the cubic
      minimizer is closer to x than the quadratic (secant) one,
      the cubic one is taken, else the quadratic one is taken.
    */
    *brackt = 1;
    bound = 0;
    CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
    QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
    if (fabs(mc - *t) > fabs(mq - *t)) {
      newt = mc;
    } else {
      newt = mq;
    }
  } else if (fabs(*dt) < fabs(*dx)) {
    /*
      Case 3: a lower function value, derivatives of the
      same sign, and the magnitude of the derivative decreases.
      The cubic minimizer is only used if the cubic tends to
      infinity in the direction of the minimizer or if the minimum
      of the cubic is beyond t. Otherwise the cubic minimizer is
      defined to be either tmin or tmax. The quadratic (secant)
      minimizer is also computed and if the minimum is brackt
      then the the minimizer closest to x is taken, else the one
      farthest away is taken.
    */
    bound = 1;
    CUBIC_MINIMIZER2(mc, *x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
    QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
    if (*brackt) {
      if (fabs(*t - mc) < fabs(*t - mq)) {
        newt = mc;
      } else {
        newt = mq;
      }
    } else {
      if (fabs(*t - mc) > fabs(*t - mq)) {
        newt = mc;
      } else {
        newt = mq;
      }
    }
  } else {
    /*
      Case 4: a lower function value, derivatives of the
      same sign, and the magnitude of the derivative does
      not decrease. If the minimum is not brackt, the step
      is either tmin or tmax, else the cubic minimizer is taken.
    */
    bound = 0;
    if (*brackt) {
      CUBIC_MINIMIZER(newt, *t, *ft, *dt, *y, *fy, *dy);
    } else if (*x < *t) {
      newt = tmax;
    } else {
      newt = tmin;
    }
  }

  /*
    Update the interval of uncertainty. This update does not
    depend on the new step or the case analysis above.

    - Case a: if f(x) < f(t),
    x <- x, y <- t.
    - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
    x <- t, y <- y.
    - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
    x <- t, y <- x.
  */
  if (*fx < *ft) {
    /* Case a */
    *y = *t;
    *fy = *ft;
    *dy = *dt;
  } else {
    /* Case c */
    if (dsign) {
      *y = *x;
      *fy = *fx;
      *dy = *dx;
    }
    /* Cases b and c */
    *x = *t;
    *fx = *ft;
    *dx = *dt;
  }

  /* Clip the new trial value in [tmin, tmax]. */
  if (tmax < newt) newt = tmax;
  if (newt < tmin) newt = tmin;

  /*
    Redefine the new trial value if it is close to the upper bound
    of the interval.
  */
  if (*brackt && bound) {
    mq = *x + 0.66 * (*y - *x);
    if (*x < *y) {
      if (mq < newt) newt = mq;
    } else {
      if (newt < mq) newt = mq;
    }
  }

  /* Return the new trial value. */
  *t = newt;
  return 0;
}

static lbfgsfloatval_t owlqn_x1norm(
                                    const lbfgsfloatval_t* x,
                                    const int start,
                                    const int n
                                    )
{
  int i;
  lbfgsfloatval_t norm = 0.;

  for (i = start;i < n;++i) {
    norm += fabs(x[i]);
  }

  return norm;
}

static void owlqn_pseudo_gradient(
                                  lbfgsfloatval_t* pg,
                                  const lbfgsfloatval_t* x,
                                  const lbfgsfloatval_t* g,
                                  const int n,
                                  const lbfgsfloatval_t c,
                                  const int start,
                                  const int end
                                  )
{
  int i;

  /* Compute the negative of gradients. */
  for (i = 0;i < start;++i) {
    pg[i] = g[i];
  }

  /* Compute the psuedo-gradients. */
  for (i = start;i < end;++i) {
    if (x[i] < 0.) {
      /* Differentiable. */
      pg[i] = g[i] - c;
    } else if (0. < x[i]) {
      /* Differentiable. */
      pg[i] = g[i] + c;
    } else {
      if (g[i] < -c) {
        /* Take the right partial derivative. */
        pg[i] = g[i] + c;
      } else if (c < g[i]) {
        /* Take the left partial derivative. */
        pg[i] = g[i] - c;
      } else {
        pg[i] = 0.;
      }
    }
  }

  for (i = end;i < n;++i) {
    pg[i] = g[i];
  }
}

static void owlqn_project(
                          lbfgsfloatval_t* d,
                          const lbfgsfloatval_t* sign,
                          const int start,
                          const int end
                          )
{
  int i;

  for (i = start;i < end;++i) {
    if (d[i] * sign[i] <= 0) {
      d[i] = 0;
    }
  }
}


/* make the lua/torch side generic Tensors (including cuda tensors)
   while this lbfgs code always works on doubles */

static const void *current_torch_type    = NULL;
static const void *torch_DoubleTensor_id = NULL;
static const void *torch_FloatTensor_id  = NULL;
static const void *torch_CudaTensor_id   = NULL;

static void *parameters     = NULL;
static void *gradParameters = NULL;

#include "generic/lbfgs.c"
#include "THGenerateFloatTypes.h"

#ifdef WITH_CUDA
/* generate cuda code */
#include "generic/lbfgs.c"
#define real float
#define Real Cuda
#define TH_REAL_IS_CUDA
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef Real
#undef TH_REAL_IS_CUDA
#undef TH_GENERIC_FILE
#endif

static int nParameter = 0;
static lua_State *GL = NULL;
static lbfgs_parameter_t lbfgs_param;
static lbfgsfloatval_t *x = NULL;

static lbfgsfloatval_t evaluate(void *instance,
                                const lbfgsfloatval_t *x,
                                lbfgsfloatval_t *g,
                                const int n,
                                const lbfgsfloatval_t step)
{

  if ( current_torch_type == torch_DoubleTensor_id )
    THDoubleTensor_copy_evaluate_start(parameters, x, nParameter);
  else if ( current_torch_type == torch_FloatTensor_id )
    THFloatTensor_copy_evaluate_start(parameters, x, nParameter);
#ifdef WITH_CUDA
  else if ( current_torch_type == torch_CudaTensor_id )
    THCudaTensor_copy_evaluate_start(parameters, x, nParameter);
#endif
  /* evaluate f(x) and g(f(x)) */
  lua_getfield(GL, LUA_GLOBALSINDEX, "lbfgs");   /* table to be indexed */
  lua_getfield(GL, -1, "evaluate");              /* push result of t.x (2nd arg) */
  lua_remove(GL, -2);                            /* remove 'lbfgs' from the stack */
  lua_call(GL, 0, 1);                            /* call: fx = lbfgs.evaluate() */
  lbfgsfloatval_t fx = lua_tonumber(GL, -1);     /* return fx */

  /* incr eval counter */
  nEvaluation ++;

  if ( current_torch_type == torch_DoubleTensor_id )
    THDoubleTensor_copy_evaluate_end(g, gradParameters, nParameter);
  else if ( current_torch_type == torch_FloatTensor_id )
    THFloatTensor_copy_evaluate_end(g, gradParameters, nParameter);
#ifdef WITH_CUDA
  else if ( current_torch_type == torch_CudaTensor_id )
    THCudaTensor_copy_evaluate_end(g, gradParameters, nParameter);
#endif

  /* return f(x) */
  return fx;
}

static int cg_progress(void *instance,
                       const lbfgsfloatval_t *x,
                       const lbfgsfloatval_t *g,
                       const lbfgsfloatval_t fx,
                       const lbfgsfloatval_t xnorm,
                       const lbfgsfloatval_t gnorm,
                       const lbfgsfloatval_t step,
                       int n,
                       int k,
                       int ls)
{
  nIteration = k;
  if (verbose > 1) {
    printf("<cg()> iteration %d:\n", nIteration);
    if (verbose > 2){
      print_fxxdx(fx,x,g,n);
      printf("  + xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    }
    printf("  + nb evaluations = %d\n", nEvaluation);
  }
  return 0;
}

static int lbfgs_progress(void *instance,
                          const lbfgsfloatval_t *x,
                          const lbfgsfloatval_t *g,
                          const lbfgsfloatval_t fx,
                          const lbfgsfloatval_t xnorm,
                          const lbfgsfloatval_t gnorm,
                          const lbfgsfloatval_t step,
                          int n,
                          int k,
                          int ls)
{
  nIteration = k;
  if (verbose > 1) {
    printf("<lbfgs()> iteration %d:\n", nIteration);
    if (verbose > 2){
      print_fxxdx(fx,x,g,n);
      printf("  + xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    }
    printf("  + nb evaluations = %d\n", nEvaluation);
  }
  return 0;
}

int lbfgs_init(lua_State *L){
  /* initialize the parameters for the L-BFGS optimization */
  lbfgs_parameter_init(&lbfgs_param);
  lbfgs_param.max_evaluations  = lua_tonumber(L, 3);
  lbfgs_param.max_iterations   = lua_tonumber(L, 4);
  lbfgs_param.max_linesearch   = lua_tonumber(L, 5);
  lbfgs_param.orthantwise_c    = lua_tonumber(L, 6);
  lbfgs_param.linesearch       = lua_tonumber(L, 7);
  /* get verbose level */
  verbose = lua_tonumber(L,8);
  /* now load the common parameter and gradient vectors */
  init(L);

   return 0;
}

int cg_init(lua_State *L){
  /* initialize the parameters for the L-BFGS optimization */
  lbfgs_parameter_init(&lbfgs_param);
  lbfgs_param.max_evaluations  = lua_tonumber(L, 3);
  lbfgs_param.max_iterations = lua_tonumber(L, 4);
  lbfgs_param.max_linesearch = lua_tonumber(L, 5);
  lbfgs_param.momentum       = lua_tonumber(L, 6);
  lbfgs_param.linesearch     = lua_tonumber(L, 7);
  /* get verbose level */
  verbose = lua_tonumber(L,8);
  /* now load the common parameter and gradient vectors */
  init(L);

  return 0;
}

int init(lua_State *L) {
  /* get params from userspace */
  GL = L;

  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");
#ifdef WITH_CUDA
  torch_CudaTensor_id = luaT_checktypename2id(L, "torch.CudaTensor");
#endif
  /* copy lua function parameters of different types into this namespace */
  void *src;
  if (src = luaT_toudata(L,1,torch_DoubleTensor_id))
    {
      parameters     = luaT_checkudata(L, 1, torch_DoubleTensor_id);
      gradParameters = luaT_checkudata(L, 2, torch_DoubleTensor_id);
      nParameter = THDoubleTensor_nElement((THDoubleTensor *) parameters);
      current_torch_type = torch_DoubleTensor_id;
    }
  else if (src = luaT_toudata(L,1,torch_FloatTensor_id))
    {
      parameters     = luaT_checkudata(L, 1, torch_FloatTensor_id);
      gradParameters = luaT_checkudata(L, 2, torch_FloatTensor_id);
      nParameter = THFloatTensor_nElement((THFloatTensor *) parameters);
      current_torch_type = torch_FloatTensor_id;
    }
#ifdef WITH_CUDA
  else if (src = luaT_toudata(L,1,torch_CudaTensor_id))
    {
      parameters     = luaT_checkudata(L, 1, torch_CudaTensor_id);
      gradParameters = luaT_checkudata(L, 2, torch_CudaTensor_id);
      nParameter = THCudaTensor_nElement((THCudaTensor *) parameters);
      current_torch_type = torch_CudaTensor_id;
    }
#endif
  else
    {
      luaL_typerror(L,1,"torch.*Tensor");
    }

  /* parameters for algorithm */
  nEvaluation = 0;
  x = lbfgs_malloc(nParameter);

  /* dispatch the copies */
  if ( current_torch_type == torch_DoubleTensor_id )
    THDoubleTensor_copy_init(x,(THDoubleTensor *)parameters,nParameter);
  else if ( current_torch_type == torch_FloatTensor_id )
    THFloatTensor_copy_init(x,(THFloatTensor *)parameters,nParameter);
#ifdef WITH_CUDA
  else if ( current_torch_type = torch_CudaTensor_id )
    THCudaTensor_copy_init(x,(THCudaTensor *)parameters,nParameter);
#endif


  /* done */
  return 0;
}

int clear(lua_State *L) {
  /* cleanup */
  lbfgs_free(x);
  return 0;
}
int print_fxxdx (lbfgsfloatval_t fx,
                 const lbfgsfloatval_t *x,
                 const lbfgsfloatval_t *dx,
                 int n){
  printf("  + fx = %f\n", fx);
  if (nParameter > 10) {
    printf("  +  x = [%f, %f, %f, ..., %f, %f ,%f]\n",
           x[0],x[1],x[2],x[n-3],x[n-2],x[n-1]);
    printf("  + dx = [%f, %f, %f, ..., %f, %f , %f]\n",
           dx[0],dx[1],dx[2],dx[n-3],dx[n-2],dx[n-1]);
  } else {
    int i;
    printf("  +  x = [%f", x[0]);
    for (i=1; i<n;i++) {
      printf(", %f", x[i]);
    }
    printf("]\n");
    printf("  + dx = [%f", dx[0]);
    for (i=1; i<n;i++) {
      printf(", %f", dx[i]);
    }
    printf("]\n");
  }
}


int lbfgs_run(lua_State *L) {
  /* check existence of x */
  if (!x) {
    THError("lbfgs.init() should be called once before calling lbfgs.run()");
  }
  /* reset our counter */
  nEvaluation = 0;

  /*  Start the L-BFGS optimization; this will invoke the callback functions */
  /*  evaluate() and progress() when necessary. */
  static lbfgsfloatval_t fx;
  int ret = lbfgs(nParameter, x, &fx, evaluate, lbfgs_progress, NULL, &lbfgs_param);

  /*  verbose */
  if (verbose) {
    printf("<lbfgs_run()> batch optimized after %d iterations\n", nIteration);
    printf("  + nb evaluations = %d\n", nEvaluation);
    if (verbose > 1){
      print_fxxdx(fx,x,gradParameters,nParameter);      
      print_linesearch_type(lbfgs_param.linesearch);
    }
  }

  /*  return current error */
  lua_pushnumber(L, fx);
  return 1;
}

int cg_run(lua_State *L) {
  /* check existence of x */
  if (!x) {
    THError("cg.init() should be called once before calling cg.run()");
  }
  /* reset our counter */
  nEvaluation = 0;

  /*  Start the CG optimization; this will invoke the callback functions */
  /*  evaluate() and progress() when necessary. */
  static lbfgsfloatval_t fx;
  int ret = cg(nParameter, x, &fx, evaluate, cg_progress, NULL, &lbfgs_param);

  /*  verbose */
  if (verbose) {
    printf("<cg_run()> batch optimized after %d iterations\n", nIteration);
    printf("  + nb evaluations = %d\n", nEvaluation);
    printf("  + linesearch = %d , momentum = %d\n",
           lbfgs_param.linesearch, lbfgs_param.momentum);
    if (verbose > 1){
      print_fxxdx(fx,x,gradParameters,nParameter);
      print_linesearch_type(lbfgs_param.linesearch);
    }
  }

  /*  return current error */
  lua_pushnumber(L, fx);
  return 1;
}

static const struct luaL_Reg cg_methods__ [] = {
  /* clear is the same method */
  {"init",  cg_init},
  {"clear", clear},
  {"run",   cg_run},
  {NULL, NULL}
};

static const struct luaL_Reg lbfgs_methods__ [] = {
  {"init",  lbfgs_init},
  {"clear", clear},
  {"run",   lbfgs_run},
  {NULL, NULL}
};

DLL_EXPORT int luaopen_liblbfgs(lua_State *L)
{
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  luaT_pushmetaclass(L, torch_DoubleTensor_id);
  luaT_registeratname(L, lbfgs_methods__, "lbfgs");
  lua_pop(L,1);

  luaL_register(L, "lbfgs", lbfgs_methods__);

  luaL_register(L, "cg", cg_methods__);

  return 1;
}
