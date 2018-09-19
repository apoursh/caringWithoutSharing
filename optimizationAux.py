# Optimization helper functions 

import numpy as np 
from sklearn.metrics import log_loss
from numpy import sum, maximum, exp, log, log1p
from scipy.optimize import fmin_l_bfgs_b as bfgs
from scipy.optimize.lbfgsb import _minimize_lbfgsb
from scipy.optimize.optimize import MemoizeJac, wrap_function
from scipy.optimize._lbfgsb import setulb
from numpy.linalg import cholesky
import pdb
from scipy.linalg import get_blas_funcs
from numpy.core._methods import _sum
from numpy.core import umath
from itertools import imap,starmap,izip
from operator import mul
#from profilehooks import profile

from scipy.special import expit

#gemm = get_blas_funcs("dgemm", [X, Y])
gemm = get_blas_funcs("gemm")
umr_sum = umath.add.reduce
add     = np.add
sub     = np.subtract
dot     = np.core.multiarray.dot
#dot = get_blas_funcs('gemm', (A, B))
trans   = np.transpose
div     = np.divide
asarray = np.asarray
zeros   = np.zeros
int32   = np.int32
float64 = np.float64
array   = np.array
mult    = np.multiply

# ADMM helpers 
#@nb.jit(nb.types.Tuple((nb.f8, nb.f8[:,:])) (nb.f8[:], nb.f8[:,:], nb.f8[:,:],nb.f8[:,:],nb.f8, nb.i8), locals={'v':nb.f8[:,:],
#  'eCx': nb.f8[:,:], 'f': nb.f8, 'g': nb.f8[:,:], 'y':nb.f8[:,:]})
def l2_log(x, C, z, u, rho, n):
  """ assumes C is an numpy array"""
  x = x.reshape(n,1)
  v = sub(add(x,u), z)
  eCx = exp(dot(C,x))
#  eCx = exp(gemm(alpha=1.0, a=C, b=x, trans_a=False, trans_b=False))

  f = _sum(log1p(eCx)) + 0.5 * rho * _sum(mult(v, v))
  g = dot(C.T, div(eCx, (add(1, eCx)))) +  rho * v
  return f, g

def bfgs_update(C, u, z, rho, x0):
  args = (C, z, u, rho) 
  # bfgs(func, x0, args, bounds, callback)
  return bfgs(l2_log, x0, args=args, bounds=None, callback=callback)

def bfgs_gutted(C, u, z, rho, x0):
   args = (C, z, u, rho) 
   fun = MemoizeJac(l2_log)
   jac = fun.derivative
   return _minimize_lbfgsb(fun, x0, args, jac, None, None, 10,
      2.2204460492503131e-09, 1e-5, 1e-8, 15000, 15000, -1, None, 20)['x']

def bfgs_more_gutted(C, u, z, rho, x, n):
  max_iter = 15000
#  fun = MemoizeJac(l2_log)
#  jac = fun.derivative
  m = 10
#  epsilon = 1e-8
#  pgtol = 1e-5
#  factr = 1e-7#ftol / np.finfo(float).eps
  x = x.ravel()
#  def func_and_grad(x):
#    f = fun(x, C, z, u, rho, n)
#    g = jac(x, C, z, u, rho, n)
#    return f, g
  nbd = zeros(n, int32)
  low_bnd = zeros(n, float64)
  upper_bnd = zeros(n, float64)

#  x = array(x0, float64)
#  f = array(0.0, float64)
  f = 0.0
  g = zeros((n,), float64)
#  wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, float64)
  wa = zeros(20*n + 5*n + 1180, float64)
  iwa = zeros(3*n, int32)
  task = zeros(1, 'S60')
  csave = zeros(1, 'S60')
  lsave = zeros(4, int32)
  isave = zeros(44, int32)
  dsave = zeros(29, float64)
  
  
  task[:] = 'START'

  n_iterations = 0
  while 1:
    setulb(m, x, low_bnd, upper_bnd, nbd, f, g, 1e-7,
        1e-5, wa, iwa, task, -1, csave, lsave,isave, dsave, 20)
    #task_str = task.tostring()
#    if task_str.startswith(b'FG'):
    task_str = task[0][:2]
    if task_str == 'FG':
      # reduce call time put the code here 
      y = x.reshape(n,1)
      v = sub(add(y,u),z)
      eCx = exp(dot(C,y))
      rhov = rho * v
#      fobj = umr_sum(log1p(eCx), None, None, None, False) 
#      f = fobj + 0.5 * umr_sum(mul(rhov,v), None, None, None, False)
      f = umr_sum(log1p(eCx), None, None, None, False) + 0.5 * umr_sum(mul(rhov,v), None, None, None, False)
      g = dot(C.T, div(eCx, (add(1, eCx)))) + rhov
    elif task_str == 'NE':
#    elif task_str.startswith(b'NEW_X'):
      n_iterations += 1
      if n_iterations >= max_iter:
        return x
    else:
      return x
      #break
  
  
def shrinkage(a, kappa):
  return maximum(0, a - kappa) - maximum(0, -a - kappa)

def l1_OLS(A, b, lam, x, z):
  return 0.5 * sum((A.dot(x) - b) ** 2 ) + lam * np.norm(z,1)

def lasso_admm_cholesky(A, rho):
  n, p = A.shape
  if n > p:
    chol = cholesky(A.T.dot(A) + rho * np.eye(p, dtype=np.float))
  else:
    chol = cholesky(np.eye(n, dtype=np.float) + 1.0 / rho * A.dot(A.T))
  return chol

def callback(x): 
  pass 




