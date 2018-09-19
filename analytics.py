# Armin Pourshafeie

#TODO write a generator that takes the chromosome and spits out data. do the regression in parallel 
#TODO documentation
# Running the gwas
import logging
import numpy as np
import gzip, h5py, os, re, gc, tqdm
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import sklearn.decomposition as decomp 
from scipy.linalg import svd
from scipy.stats import chi2 
from scipy.sparse.linalg import eigsh as eig 
import mkl
from optimizationAux import *
from plinkio import plinkfile
# Careful here, eigh uses https://software.intel.com/en-us/mkl-developer-reference-c-syevr behind the hood
# so it can be significantly slower 
from numpy.core import _methods
from sklearn.utils.extmath import randomized_svd, svd_flip
import time, sys
from corr import nancorr, corr, HweP
from numpy.linalg import inv as inverse
from numpy.core import umath as um
from numpy import mean, isnan
from sklearn.metrics import log_loss


#from numpy.core import umath as um
#umr_maximum = um.maximum.reduce
umr_sum = um.add.reduce
maximum = np.maximum
add     = np.add
_mean   = _methods._mean
_sum    = _methods._sum
sub     = np.subtract 
div     = np.divide 
chi2sf  = chi2.sf
sqrt    = np.sqrt

mean = np.mean

kExactTestBias = 0.00000000000000000000000010339757656912845935892608650874535669572651386260986328125;
kSmallEpsilon = 0.00000000000005684341886080801486968994140625;
kLargeEpsilon = 1e-7



class DO(object):
  """This object represents each data owner. It can compute statistics in a 
  centralized manner on it's own data, or if it has a centeral hub associated with it it can 
  communicate with the center"""
  def __init__(self, store_name, center=None):
    self.store_name = store_name
    self.center = center

    with h5py.File(self.store_name) as store:
      self.has_local_AF = ('has_local_AF' in store.attrs and 
          store.attrs['has_local_AF'])
      self.normalized = ('normalized' in store.attrs and 
          store.attrs['normalized'])
      self.n = store['meta/Status'].shape[0]
    self.current_X = None
    self.current_Y = None
    self.load_snp  = True 

  def clear_tmp(self):
    self.current_X = None
    self.current_Y = None
  
  def clear_tmpX(self):
    self.current_X = None
    self.X         = None

  def clear_tmpY(self):
    self.current_Y = None

  def count(self, exclude=['meta']):
    with h5py.File(self.store_name, 'r')  as store:
      chroms = [ chrom for chrom in store.keys() if chrom not in exclude ]
      c = 0
      for chrom in chroms:
        c += len(store[chrom].keys())
    self.p = c
    return c


  def likelihood(self, beta, verbose=False):
    """log loss. If beta is a matrix. Verbose refers to when beta is a matrix and not just a vector"""
    y_model= 1.0 / (1 + np.exp(-self.X.dot(beta)))
    if not verbose:
      return log_loss((self.current_Y+1)/2, y_model, normalize=False, labels=[0,1])
    else:
      return np.array([log_loss((self.current_Y+1)/2, y_pred, normalize=False, labels=[0,1]) for y_pred in y_model.T])

  def local_missing_filter(self, t_missing):
    n = float(self.n)
    def _filter(name, node):
      if isinstance(node, h5py.Dataset) and node.parent.name != '/meta':
        if node.attrs['local_missing']/n > t_missing:
          node.attrs['local_filter'] = True

    with h5py.File(self.store_name, 'a') as f:
      f.visititems(_filter)

  def local_AF_filter(self, t_AF):
    def _filter(name, node):
      if isinstance(node, h5py.Dataset) and node.parent.name != '/meta':
        if 'local_filter' in node.attrs and node.attrs['local_filter']:
          # already filtered 
          return 
        local_AF = node.attrs['local_AF']
        if local_AF + kLargeEpsilon < t_AF or local_AF - kLargeEpsilon > (1-t_AF):
          node.attrs['local_filter'] = True
    with h5py.File(self.store_name, 'a') as f:
      f.visititems(_filter)

  def local_HWE_filter(self, t_hwe):
    def _filter(name, node):
      if isinstance(node, h5py.Dataset) and node.parent.name != '/meta':
        if 'local_filter' not in node.attrs or not node.attrs['local_filter']:
          v = node.value
          vhet = np.sum(v==1)
          vhr  = np.sum(v==0)
          vha = np.sum(v==2)
          hwe = HweP(vhet, vhr, vha, 0)
          if hwe < t_hwe:
            node.attrs['local_filter'] = True
    with h5py.File(self.store_name, 'a') as f: 
      f.visititems(_filter)

  def local_LD_filter(self, t_ld, win_sz, step_sz=None):
    def pruner(chrom, threshold, window):
      window.shape = (1, window.shape[0])
      to_delete = set()
      n = window.shape[1]
      sumLinT, sumSqT, crossT = self.corr_data([chrom], window)
      MAF = self.get_MAF(chrom, window[0], global_freq=False)
      corrT = corr(sumLinT, sumSqT, crossT)
      while (1):
        for i, snp1 in enumerate(window[0,:]):
          if snp1 in to_delete:
            continue
          else:
            for j in range(i+1, n):
              if window[0][j] in to_delete:
                continue
              elif corrT[i,j]**2 > t_ld:
                if MAF[i] > MAF[j] * (1.0 + kLargeEpsilon):   #somewhat similar to what plink does
                  to_delete.add(snp1)
                else: 
                  to_delete.add(window[0][j])
                break
        remaining = np.array([i for i,snp in enumerate(window[0]) if snp not in to_delete])
        r2 = corrT[remaining,:][:,remaining]
        if np.max(r2**2) < t_ld:
          break
      return to_delete

    if step_sz is None:
      step_sz = int(win_sz/2)
    with h5py.File(self.store_name, 'a') as f: 
      for chrom in f.keys():
        if chrom == 'meta':
          continue 
        # Get snps that pass the allele frequency threshold
        dset = f[chrom]
        allsnps = np.array(self.snps_present(chrom))
        snps = np.sort(np.array([int(snp) for snp in allsnps if ('local_filter' 
          not in dset[snp].attrs or not dset[snp].attrs['local_filter'])]))
        del allsnps
        win_sz = min(snps.shape[0], win_sz)
        finished, winstart  = False, 0
        highLD, to_delete = set(), set()
        while not finished:
          winend = winstart + win_sz
          if winend >= len(snps):
            finished = True 
            winend = len(snps)
          window = snps[winstart:winend] #preliminary window 
          window = np.sort(np.array(list(set(window) - to_delete)))#[:win_sz]
          to_delete = pruner(chrom, t_ld, window)
          highLD = highLD.union(to_delete)
          winstart += step_sz
        # Mark highLD 
        for snp in highLD:
          dset[str(snp)].attrs['local_filter'] = True


  def clean_by_local_filter(self,chrom=None, keepset=set()):
    with h5py.File(self.store_name, 'a') as f: 
      if chrom is None:
        for chrom in f.keys():
          if chrom != 'meta':
            dset = f[chrom]
            for snp in dset:
              if 'local_filter' in dset[snp].attrs:
                del dset[snp]
      else: 
        dset = f[chrom]
        for snp in dset: 
          if snp not in keepset:
            del  dset[snp]



  def locally_unfiltered(self, chrom):
    present = set()
    def _counter(name, node):
      if 'local_filter' not in node.attrs:
        present.add(name)
    with h5py.File(self.store_name, 'r') as f:
      dset = f[chrom]
      dset.visititems(_counter)
    return present

  def AF_filter(self, threshold, chrom_dset):
    return [i for i in chrom_dset if chrom_dset[i].attrs['AF'
      ]>=threshold and chrom_dset[i].attrs['AF'] <= 1-threshold]
  
  def snps_present(self, chrom_dset):
    return [i for i in chrom_dset]

  def pruning(self, threshold, Af_threshold, win_sz, step_sz=None):
    """Threshold is for rsquared and win_sz is in number of snps"""
    def pruner(dset, threshold, window):
      to_delete = set()
      for i, snp in enumerate(window):
        if snp in to_delete:
          continue
        else:
          snpi = dset[str(snp)].value
          for j in range(i+1, len(window)):
            if window[j] in to_delete:
              continue
            elif np.cov(snpi, dset[str(window[j])].value)[0,1]**2 > threshold: # use only with normalzied data
              to_delete.add(window[j])
      return to_delete

    if step_sz == None:
      step_sz = int(win_sz/4)
    with h5py.File(self.store_name, 'a') as readfp:
      for chrom in readfp.keys():
        if chrom == 'meta':
          continue
        logging.info('--Pruning chrom: ' + chrom)
        dset = readfp[chrom]
        #snps = np.sort(np.array(dset.keys()).astype(int))
        snps = np.sort(np.array(self.AF_filter(Af_threshold, dset))).astype(int)
        win_sz = min(snps.shape[0], win_sz)
        finished, winstart, winend = False, 0, win_sz
        highLD = set()
        while not finished:
          winend = winstart + win_sz
          if winend >= len(snps) - 1:
            finished = True 
            winend = len(snps) - 1
          window = snps[winstart:winend]
          window = np.sort(np.array(list(set(window) - highLD)))
          to_delete = pruner(dset, threshold, window)
          highLD = highLD.union(to_delete)
          winstart += step_sz
        toKeep = set(snps) - highLD
        logging.debug("----Keeping {} snps after AF/LD pruning".format(len(toKeep)))
        for snp in toKeep: 
          dset[str(snp)].attrs['prune_selected'] = True

  def local_pca(self, n_components=None, chroms=None):
    with h5py.File(self.store_name, 'r') as store:
      if chroms is None:
        chroms = [group for group in store if group != 'meta']
      chorms = sorted(chroms, key=lambda x: int(x))
      to_PCA = []
      for chrom in chroms:
        dset = store[chrom]
        all_snps = sorted(dset.keys(), key=lambda x:int(x))
        for snp in all_snps:
          if 'local_filter' not in dset[snp].attrs or not dset[snp].attrs['local_filter']:
            val = (dset[snp].value.astype(np.float32) - 2*dset[snp].attrs['local_AF'])/dset[snp].attrs['local_sd']
            val[np.isnan(val)] = 0
            to_PCA += [list(val)]
      to_PCA = np.array(to_PCA).T
      #to_PCA = 1.0/self.n * to_PCA.T
    #pca = PCA(n_components=n_components)
    #pca.fit(to_PCA)
    N = to_PCA.shape[0]
    logging.info("-pca size is {}".format(to_PCA.shape))
    u, sigma, vt = randomized_svd(to_PCA, n_components, transpose=False)
    u,vt = svd_flip(u, vt, u_based_decision=False)
    with h5py.File(self.store_name) as store:
      dset = store['meta']

      pca_u = dset.require_dataset('pca_u_local', shape=u.shape, dtype=np.float32)
      pca_u[:,:] = u
      pca_sigma = dset.require_dataset('pca_sigma_local', shape=sigma.shape, dtype=np.float32)
      pca_sigma[:] = sigma
      pca_v = dset.require_dataset('pca_vt_local', shape=vt.shape, dtype=np.float32)
      pca_v[:] = vt


  def local_regression(self, numPCs, chrom):
    snps = sorted(self.dataset_keys(chrom), key=lambda x:int(x))
    model = LogisticRegression(fit_intercept=False, C=1e5)
    X = np.empty((self.n, numPCs+1))
    betas = np.empty((len(snps), 1))
    pvals_local = np.empty_like(betas)
    standard_error = np.empty_like(betas)
    V = np.matrix(np.zeros(shape = (X.shape[0], X.shape[0])))
    with h5py.File(self.store_name, 'r') as store:
      X[:,1:] = store['meta/pca_u_local'].value[:, :numPCs]
      X[:,1:] /= np.std(X[:,1:], axis=0)
      Y = store['meta/Status']

      dset = store[chrom]
      # Unfortunately, everything is normalized, so we need to undo that
      for i, snp_id in enumerate(snps):
        snp = dset[snp_id]
        local_sd = snp.attrs['local_sd']
        if local_sd == 0.0:
          pvals_local[i,0] = np.nan
          standard_error[i,0] = np.nan
          betas[i,0] = np.nan
        else:
          snpv = snp.value
          #Normalize with local values
          snpv -= 2*snp.attrs['local_AF']
          snpv /= local_sd
          snpv[np.isnan(snpv)] = 0
          X[:,0] = snpv
          model.fit(X, Y)
          beta = model.coef_
          betas[i, 0] = beta[0,0]
          # generate local pvalues
          expVal = np.exp(X.dot(beta.T))
          ymodel = expVal/(1+expVal)
          np.fill_diagonal(V, np.multiply(ymodel, 1-ymodel))
          F = X.T * V * X
          z = (beta/sqrt(np.diag(inverse(F))).reshape(1,numPCs+1))
          z *= z
          pvals_local[i,0] = chi2sf(z,1)[0,0]
          standard_error[i,0] = sqrt(np.diag(inverse(F))).reshape(1, numPCs+1)[0,0]
      return betas, standard_error, pvals_local


  def compute_local_AF(self):
    def __compute_AF(name, node):
      if isinstance(node, h5py.Dataset) and node.parent.name != '/meta':
        vals = node.value
        vals[vals == 3] = np.nan
        node[...] = vals
        node.attrs['local_AF'] = np.nanmean(node) / 2.
        node.attrs['n']        = node.len()
        node.attrs['local_sd'] = np.nanstd(node)
        if self.center is None: 
          node.attrs['AF']     = node.attrs['local_AF']
          node.attrs['sd']     = node.attrs['local_sd']

    logging.info("-Computing local allele frequencies")
    if self.has_local_AF:
      logging.info("--Allele frequencies have already been computed")
      return
    with h5py.File(self.store_name, 'a') as f:
      f.visititems(__compute_AF)
      self.has_local_AF = True
      f.attrs['has_local_AF'] = True

  def impute(self):
    """Use after centering the data. This simply replaces Nan's with 0"""
    def _imputer(name, node):
      if isinstance(node, h5py.Dataset) and node.parent.name != '/meta':
        vals = node.value
        AF = node.attrs['AF']
        vals[np.isnan(vals)] = 0 #(np.round(2*AF) - AF) / node.attrs['sd']
        node[...] = vals

    with h5py.File(self.store_name, 'a') as f: 
      f.visititems(_imputer)


# define a class that inherits from above for the group that has centers 

class Decentralized_DO(DO):
  """Data owner that can aid in computation of aggregate statistics"""
  
  def group_keys(self):
    with h5py.File(self.store_name, 'r') as f:
      return f.keys()
  
  def dataset_keys(self, grp):
    with h5py.File(self.store_name, 'r') as f:
      dset = f[grp]
      return dset.keys()

  def report_local_AF(self,chrom):
    AF_dic = {}
    def _report_AF(name, node):
      AF_dic[name] = node.attrs['local_AF'], node.attrs['local_sd'], self.n -  node.attrs['local_missing']

    with h5py.File(self.store_name, 'r') as f: 
      dset = f[chrom]
      dset.visititems(_report_AF)
    return AF_dic


  def report_SD(self, chrom):
    SD_dic = {}
    def _report_SD(name, node): 
      vals = node.value - 2 * node.attrs['AF']
      node[...] = vals
      SD_dic[name] = np.sqrt(np.nansum(node.value**2)), np.sum(~np.isnan(node.value))
    
    with h5py.File(self.store_name, 'a') as f: 
      dset = f[chrom]
      dset.visititems(_report_SD)
    return SD_dic

  def normalize(self, chrom):
    def _normalizer(name, node):
      val = node.value/node.attrs['sd']
      node[...] = val

    with h5py.File(self.store_name, 'a') as f:
      dset = f[chrom]
      dset.visititems(_normalizer)


  def report_local_missing_rates(self, chrom):
    MR_dic = {}
    def _report_missing_rate(name, node):
      if 'local_missing' not in node.attrs: 
        print(name)
      MR_dic[name] = node.attrs['local_missing']
    with h5py.File(self.store_name, 'r') as f: 
      dset = f[chrom]
      dset.visititems(_report_missing_rate)
    return MR_dic

  def report_local_counts(self, chrom):
    HWE_dic = {}
    def _report_local_counts(name, node):
      v = node.value
      HWE_dic[name] = (np.sum(v==0), np.sum(v==1), np.sum(v==2))
    with h5py.File(self.store_name, 'r') as f: 
      dset = f[chrom]
      dset.visititems(_report_local_counts)
    return HWE_dic


  def report_local_std_global_mean(self, chrom):
    std_dic = {}
    def _report_std(name, node):
      std_dic[name] = np.sqrt(np.mean((node.value - 2*node.attrs['AF'])**2))
    
    with h5py.File(self.store_name, 'r') as f: 
      dset = f[chrom]
      dset.visititems(_report_std)
    return  std_dic

  def set_local_AF(self, chrom, AF, pos):
    with h5py.File(self.store_name, 'a') as f:
      dset = f[chrom]
      if pos == 0:
        for key, value in AF.iteritems():
          dset[key].attrs['AF'] = value[0] / value[2]
      if pos == 1:
        for key, value in AF.iteritems():
          if key in dset: 
            dset[key].attrs['sd'] = np.sqrt(value[0]/value[1])
   
  def MAF_filter(self, chrom, rate):
    with h5py.File(self.store_name, 'a') as f:
      dset = f[chrom]
      for key in dset.keys():
        af = dset[key].attrs['AF']
        if af + kLargeEpsilon < rate or af - kLargeEpsilon > (1-rate):
          del dset[key]

  def HWE_filter(self, chrom, dic, rate):
    with h5py.File(self.store_name, 'a') as f:
      dset = f[chrom]
      for key, value in dic.iteritems():
        if value < rate:
          del dset[key]
        else: 
          dset[key].attrs['hwe'] = value 


  def set_missing_rate_filter(self, chrom, MR, rate):
    with h5py.File(self.store_name, 'a') as f:
      dset = f[chrom]
      for key, value in MR.iteritems():
        if value > rate:
          del dset[key]
        else:
          dset[key].attrs['missing_rate'] = value


  def give_cov(self, chroms, snps_list, cov=True):
    n = np.sum([len(item) for item in snps_list])
    with h5py.File(self.store_name, 'r') as f:
      arr = np.zeros((n, self.n))
      j = 0
      for i, chrom in enumerate(chroms):
        snps = snps_list[i]
        dset = f[chrom]
        for k in range(len(snps)):
          arr[j+k,:] = dset[str(snps[k])].value
        #arr[j:j+len(snps),:] = np.array([dset[str(item)] for item in snps])
        j += len(snps)
      if cov:
        return np.cov(arr)
      else:
        arr = arr.astype(np.float16)
        return arr.dot(arr.T)

  def corr_data(self, chroms, snps_list):
    n = np.sum(len(item) for item in snps_list)
    with h5py.File(self.store_name, 'r') as f: 
      arr = np.zeros((self.n,n), dtype=np.float32)
      j = 0 
      for i, chrom in enumerate(chroms):
        snps = snps_list[i]
        dset = f[chrom]
        for k in range(len(snps)):
          arr[:, j+k] = dset[str(snps[k])].value
        j += len(snps)
      corrmat = nancorr(arr)
    return(corrmat)

  def get_MAF(self, chrom, window, global_freq=True):
    with h5py.File(self.store_name, 'r') as f: 
      dset = f[chrom]
      vals = np.empty(len(window))
      for i, snp in enumerate(window):
        if global_freq:
          af = dset[str(snp)].attrs['AF'] 
        else:
          af = dset[str(snp)].attrs['local_AF']
        vals[i] = af if af > 0.5 else 1-af
    return vals 

  def give_cov_pca(self, chroms, n, curr_mat, weight, mult=5000): # a hack to deal with memory inefficiencies 
    #n = np.sum([len(item) for item in snps_list])
    mkl.set_num_threads(2)
    with h5py.File(self.store_name, 'r') as f:
      arr = np.zeros((n, self.n))
      j = 0
      for i, chrom in enumerate(chroms):
        dset = f[chrom]
        keyz = sorted([int(i) for i in dset.keys()])
        for k,key in enumerate(keyz):
          snp = dset[str(key)]
          value = snp.value
          #AF = snp.attrs['AF']
          #value -= 2*AF
          value[np.isnan(value)] = 0#(np.round(2*AF) - 2*AF)
          #value /= dset[str(key)].attrs['sd']
          arr[j+k,:] = value
        j += len(keyz)
      arr = arr.astype(np.float32)
      arr /= np.sqrt(weight)
      blocks = arr.shape[0]/mult
      for i in range(blocks):
        curr_mat[i*mult:(i+1)*mult,:] += arr[i*mult:(i+1)*mult,:].dot(arr.T)
      curr_mat[blocks*mult:,:] += arr[blocks*mult:,:].dot(arr.T)
  
  def give_data(self,chroms, n):
    """Should only be used to compute PCA locally for comparison's sake."""
    arr = np.empty((self.n, n))
    with h5py.File(self.store_name, 'r') as f: 
      j = 0 
      for i, chrom in enumerate(chroms):
        dset = f[chrom]
        keyz = sorted([int(i) for i in dset.keys()])
        for k, key in enumerate(keyz):
          value = dset[str(keyz[k])].value
          AF = dset[str(keyz[k])].attrs['AF']
          value[np.isnan(value)] = (np.round(2*AF) - 2*AF)  / dset[str(keyz[k])].attrs['sd']
          arr[:, j+k] = value
        j += len(keyz)

    return arr

  def give_snp_data(self, chrom, location, npcs):
    X = np.empty((self.n, npcs+1))
    with h5py.File(self.store_name, 'r') as f: 
      dset = f[chrom]
      X[:,0] = dset[str(location)].value
      y = np.sign(f["meta/Status"].value - 0.5).reshape(self.n, 1)
      X[:,1:] = f["meta/pca_u"].value[:, :npcs] * 1/0.10485152
    return X, y

  def give_moments(self, addresses):
    first_mom, second_mom = [], []
    with h5py.File(self.store_name, 'r') as f: 
      for address in addresses:
        vals = f[address].value
        first_mom.append(np.mean(vals, axis=0))
        second_mom.append(np.mean(vals ** 2, axis=0))
    return first_mom, second_mom

  def snp_loader(self, stds, npcs, covp, pos):
    """ Load the snp. Particularly useful if there are iterations"""
    with h5py.File(self.store_name, 'r') as f: 
      if self.current_Y is None: 
        X              = np.empty((self.n, covp))
        self.current_Y = np.sign(f["meta/Status"].value - 0.5).reshape(self.n, 1)
        X[:,-npcs:]    = f["meta/pca_u"].value[:, :npcs] * 1/stds
        self.current_X = X * -self.current_Y
        self.X         = X
      i = 0
      for chrom, loc in pos:
        snp = f[chrom + "/" + loc]
        # If nobody has any variation, don't bother 
        if snp.attrs['sd'] == 0:
          raise ValueError()
        val = snp.value
        val[np.isnan(val)] = 0
        self.X[:,i] = val
        i += 1

      self.current_X[:, :i] = self.X[:, :i] * -self.current_Y
    self.load_snp = False


  def run_regression(self, pos, npcs, beta, stds, logistic, covp):
    if self.load_snp: 
      self.snp_loader(stds, npcs, covp, pos)
    model = LogisticRegression(fit_intercept=False, C=1e5, warm_start=beta)
    model.fit(self.X, self.current_Y)
    return model.coef_

  def admm_update(self, pos, npcs, u, beta, rho, z0, stds,logistic, covp):
    """Runs a regularized logistic regression with a penalty that draws the answer
    closer to beta"""
    # If temp values are not set, set them up
    if self.load_snp:
      self.snp_loader(stds, npcs, covp, pos)

    return bfgs_more_gutted(self.current_X, u, beta, rho, z0, covp)

#    if logistic:
#      #x,v,d = bfgs_update(self.current_X, u, beta, rho, z0)
#      #x = bfgs_gutted(self.current_X, u, beta, rho, z0)
#      x = bfgs_more_gutted(self.current_X, u, beta, rho, z0, n)
#      return x
#    else:
#      pass
#    return x
  
  def covLogit(self, pos, beta, stds, logistic, last=True):
    """returns the variance covariance matrix for thelogistic regression
    with the provided parameters. Used for Wald pvalues"""
    if self.load_snp:
      pcov = len(beta)
      npcs = pcov - len(pos)
      self.X = np.empty((self.n, pcov))
      with h5py.File(self.store_name, 'r') as f: 
        i = 0
        for chrom, loc in pos:
          self.current_X[:, i] = f[chrom+"/"+loc].value
          i += 1
        self.X[:, i:] = f["meta/pca_u"].value[:, :npcs] * 1/stds
#    if logistic: 
    X = self.X
    expVal = np.exp(X.dot(beta))
    ymodel = expVal/(1+expVal)
    V = np.matrix(np.zeros(shape = (X.shape[0], X.shape[0])))
    np.fill_diagonal(V, np.multiply(ymodel, 1-ymodel))
    F = X.T * V * X
    # will move on so clear the load_snp flag
    if last:
      self.load_snp = True
    return F



  def update_pheno(self, phenodict):
    with h5py.File(self.store_name, 'a') as f: 
      dset = f['meta']
      ids = dset['id'].value
      phenos = [phenodict[i] for i in ids]
      dset['Status'][...] = phenos
  
  def copy_pca(self, other, local):
    if not local: 
      pca_u     = 'pca_u'
      pca_sigma = 'pca_sigma'
      pca_vt    = 'pca_v.T'
    else: 
      pca_u     = 'pca_u_local'
      pca_sigma = 'pca_sigma_local'
      pca_vt    = 'pca_vt_local'
    with h5py.File(self.store_name, 'a') as thisf:
      with h5py.File(other, 'r') as otherf:
        thismeta = thisf['meta']
        othermeta = otherf['meta']
        if pca_u in thismeta:
          del thismeta[pca_u]
          del thismeta[pca_sigma]
          del thismeta[pca_vt]
        pca_u_value = othermeta[pca_u].value
        us = thismeta.require_dataset(pca_u, shape=pca_u_value.shape, dtype=np.float32)
        us[:] = pca_u_value 
        del pca_u_value
        pca_sigmas = othermeta[pca_sigma].value
        ss = thismeta.require_dataset(pca_sigma, shape=pca_sigmas.shape, dtype=np.float32)
        ss[:] = pca_sigmas
        del pca_sigmas
        pca_vT = othermeta[pca_vt].value
        vs = thismeta.require_dataset(pca_vt, shape=pca_vT.shape, dtype=np.float32)
        vs[:] = pca_vT
        del pca_vT



  def record_centralized_pca(self, sigma, Us):
    with h5py.File(self.store_name, 'a') as f: 
      dset = f['meta']
      if 'Centralized_PCA_sigma' in dset:
        del dset['Centralized_PCA_sigma']
        del dset['PCA_Us_Centralized']
      first = dset.require_dataset('Centralized_PCA_sigma', shape=sigma.shape, dtype=np.float32)
      first[:] = sigma
      pca_components = dset.require_dataset('PCA_Us_Centralized', shape = Us.shape, dtype=np.float32)
      pca_components[:] = Us


  def AF_filter(self, threshold, chrom):
    with h5py.File(self.store_name, 'r') as f:
      dset = f[chrom]
      return super(Decentralized_DO, self).AF_filter(threshold, dset)
  
  def snps_present(self, chrom):
    with h5py.File(self.store_name, 'r') as f: 
      dset = f[chrom]
      return super(Decentralized_DO, self).snps_present(dset)

  def tag_snps(self, chrom, keys, attr_tag, value):
    with h5py.File(self.store_name, 'a') as f:
      dset = f[chrom]
      for key in keys:
        dset[str(key)].attrs[attr_tag] = value
  
  def delete_snps(self, chrom, keys):
    with h5py.File(self.store_name, 'a') as f:
      dset = f[chrom]
      for key in keys:
        del dset[str(key)]

  def passed_LD(self, chrom):
    indicies = []
    def was_selected(name, node):
      if 'prune_selected' in node.attrs:
        indicies.append(name)
    
    with h5py.File(self.store_name, 'r') as f:
      dset = f[chrom]
      dset.visititems(was_selected)
    return sorted(indicies, key=lambda x: int(x))

  def store_eigs(self, sigma, v, chroms):
    """Computes U's given the centralized sigma and V. Stores all the variables"""
    with h5py.File(self.store_name, 'a') as store:
      dset = store['meta']
      pca_sigma = dset.require_dataset('pca_sigma', shape=sigma.shape,
          dtype = np.float16)
      sigma = np.sqrt(sigma)
      pca_sigma[:] = sigma
      
      inv_sig = sigma.copy()
      inv_sig[inv_sig > 0] = 1.0/inv_sig[inv_sig > 0]
      # this part can be done for small groups at a time to save memory
      n =  self.count()#np.sum([len(item) for item in snps_list])
      arr = np.zeros((self.n, n))
      j = 0
      for i, chrom in enumerate(chroms):
        dset = store[chrom]
        snps = sorted([int(i) for i in dset.keys()])
        for k, key in enumerate(snps):
          val = dset[str(key)].value
          # It is already normalized and centered
#          AF = dset[str(key)].attrs['AF']
#          val -= 2*AF
          val[np.isnan(val)] = 0#(np.round(2*AF) - 2*AF) #/ dset[str(snps[k])].attrs['sd']
          arr[:, j+k] = val.T
        #arr[:, j:j+len(snps)] = np.array([dset[str(item)] for item in snps]).T
        j += len(snps)
      u = arr.dot(v.T).dot(np.diag(inv_sig))
      u, v = svd_flip(u, v, u_based_decision=False)
      dset = store['meta']
      pca_vt = dset.require_dataset('pca_v.T', shape=v.shape, dtype=np.float32)
      pca_vt[:,:] = v
      pca_u = dset.require_dataset('pca_u', shape=u.shape, dtype=np.float32)
      pca_u[:,:] = u

  def set_normalized(self, value):
    with h5py.File(self.store_name, 'a') as store:
      store.attrs['normalized'] = value

  def compute_local_missing_rates(self):
    def __compute_missing_rate(name, node):
      if isinstance(node, h5py.Dataset) and node.parent.name != '/meta':
        node.attrs['local_missing'] = np.sum(node.value==3)

    logging.info("-Computing local missing rates")
    with h5py.File(self.store_name, 'a') as f:
      f.visititems(__compute_missing_rate)



class Center(object):
  """The central hub that drives and requires particular computations from each node."""
  def __init__(self, store_names, n_cores=1):
    self.store_names = store_names
    self.nDOs = len(store_names)
    self.ncores = n_cores
    self.DOs = [Decentralized_DO(s_name, self) for s_name in self.store_names]
    self.keys = self.DOs[0].group_keys()
    self.n   = sum([item.n for item in self.DOs])
    logging.info("- Setup center with {} DOs and {} individuals". format(
      self.nDOs, self.n))

  def loci_missing_rate_filter(self, rate):
    for DO in self.DOs: 
      DO.compute_local_missing_rates()
    for chrom in self.keys:
      if chrom != 'meta':
        logging.info("Consensus missing rate computation on chrom: {}".format(chrom))
        MR = add_dict()
        MR.set_key_values(self.DOs[0].dataset_keys(chrom), 0)
        for DO in self.DOs:
          update_dic = DO.report_local_missing_rates(chrom)
          MR.update(update_dic, 1.0)
        for DO in self.DOs: 
          DO.set_missing_rate_filter(chrom, MR, rate * self.n)


  def MAF_filter(self, rate):
    """Computes local and consensus AF, sd. 
    Removes loci below the specified MAF"""
    def AF_wrapper(DO):
      DO.compute_local_AF()
    #with Pool(self.ncores) as pool:
      #pool.map(AF_wrapper , self.DOs)
    for DO in self.DOs:
      AF_wrapper(DO)
    for chrom in self.keys:
      if chrom != 'meta':
        logging.info("---Consensus AF computation on chrom: {}".format(chrom))
        AF = add_dict()
        AF.set_key_values(self.DOs[0].dataset_keys(chrom),[0,0,0])
        for DO in self.DOs:
          update_dic = DO.report_local_AF(chrom)
          AF.update(update_dic, 1.0, 0) 
        # update the overall AF
        for DO in self.DOs: 
          DO.set_local_AF(chrom, AF, 0)
          if rate is not None:
            DO.MAF_filter(chrom, rate)

  def compute_std(self, chrom):
    if chrom != 'meta':
      logging.info("--consensus SD computation on chrom: {}".format(chrom))
      SD = add_dict()
      SD.set_key_values(self.DOs[0].dataset_keys(chrom), [0,0])
      for DO in self.DOs:
        update_dic = DO.report_SD(chrom)
        SD.update(update_dic, 1.0, 1) #TODO This is a colossal fuck up (AF,SD, HWE). All of this shit needs to be done by passing counts As the sufficient statistics. but too late for that shit now. will clean up later
      for DO in self.DOs: 
        DO.set_local_AF(chrom, SD, 1)


  def normalize(self):
    for chrom in self.keys:
      if chrom != 'meta':
        logging.info("--normalizing chrom: {}".format(chrom))
        self.compute_std(chrom)
        for DO in self.DOs: 
          DO.normalize(chrom)


  def HWE_filter(self, rate):
    for chrom in self.keys:
      if chrom != 'meta':
        logging.info("-HWE computation on chrom: {}".format(chrom))
        HWE = add_dict()
        HWE.set_key_values(self.DOs[0].dataset_keys(chrom),np.array([0,0,0]))
        for DO in self.DOs:
          update_dic = DO.report_local_counts(chrom)
          HWE.update(update_dic, 1.0)
        for key, value in HWE.iteritems(): 
          hwe = HweP(int(value[1]), int(value[0]), int(value[2]), 0 )
          HWE[key] = hwe
        for DO in self.DOs: 
          DO.HWE_filter(chrom, HWE, rate)
  
  def HWE_test(self, homor, het, homoa):
    """HWE test (midpoint test). Other versions of HWE filter can be impelemented with the same information. 
    This implementation should match PLINK1.9's implementation."""
    homc = max(homor, homoa)
    homr = min(homor, homoa)
    
    rare = 2 * homr + het
    # mid point of the distribution 
    n = (homor + het + homoa) * 2
    tail_p = (1 - kSmallEpsilon) * kExactTestBias
    centerp = 0
    lastp2, lastp1 = tailp, tailp 
    #if (obs_hets * genotypes2 > rare_copies * (genotypes2 - rare_copies)):

    mid = int(rare * (2 *  n -rare) / (2 * n))
    if (mid % 2 != rare % 2):
      mid += 1
    probs = np.zeros(1 + rare)
    probs[mid] = 1.0
    tsum = 1.0
    curr_hets = mid
    curr_homr = (rare - mid) / 2
    curr_homc = n - curr_hets - curr_homr
    
    while (curr_hets >= 2):
      probs[curr_hets - 2] = probs[curr_hets ] * (curr_hets) * (curr_hets - 1.0) / (4.0 * (curr_homr - 1.0) * (curr_homc + 1.0))
      tsum += probs[curr_hets - 2]
      curr_hets -= 2 
      curr_homr += 1
      curr_homc += 1
      
    curr_hets = mid 
    curr_homr = (rare - mid) / 2
    curr_homc = n - curr_hets - curr_homr
    while (curr_hets <= rare -2):
      probs[curr_hets + 2] = probs[curr_hets] * 4.0 * curr_homr * curr_homc / ((curr_hets + 2.0) * (curr_hets + 1.0))
      tsum += probs[curr_hets + 2]
      curr_hets += 2
      curr_homr -= 1
      curr_homc -= 1
    
#    target = probs[het]
#    return min(1.0, np.sum(probs[probs <= target])/tsum)

    probs /= tsum
    p_hi = float(probs[het])
    for i in xrange(het + 1, rare + 1):
      p_hi += probs[i]
#    
    p_lo = float(probs[het])
    for i in xrange(het-1, -1, -1):
      p_lo += probs[i]
    p_hi_lo = 2.0 * p_hi if p_hi < p_lo else 2.0 * p_lo
    p_hwe = 0.0
    for i in xrange(0, rare + 1):
      if probs[i] > probs[het]:
        continue
      p_hwe += probs[i]
    p_hwe = 1.0 if p_hwe > 1.0 else p_hwe

    return p_hwe


  def correct_LD_prune(self, threshold, win_sz, step_sz=None):
    #TODO use local_LD_filter 
    def pruner(chrom, threshold, window):
      window.shape = (1, window.shape[0])
      to_delete = set()
      n = window.shape[1]
      sumLinT = np.zeros((n,n), dtype = np.float32)
      sumSqT = np.zeros((n,n), dtype = np.float32)
      crossT = np.zeros((n,n), dtype = np.float32)
      for DO in self.DOs: 
        sumLin, sumSq, cross = DO.corr_data([chrom], window)
        sumLinT += sumLin
        sumSqT  += sumSq
        crossT  += cross
      MAF = DO.get_MAF(chrom, window[0], global_freq=True)
      corrT = corr(sumLinT, sumSqT, crossT)
      while (1):
        for i, snp1 in enumerate(window[0,:]):
          if snp1 in to_delete:
            continue
          else:
            for j in range(i+1, n):
              if window[0][j] in to_delete:
                continue
              elif corrT[i,j]**2 > threshold:
                if MAF[i] > MAF[j] * (1.0 + kLargeEpsilon):   #somewhat similar to what plink does
                #ai = sumLin[i,j] / cross[i, j]
                #aj = sumLin[j,i] / cross[i, j]
                #majori = ai if ai > .5 else 1 - ai
                #majorj = aj if aj > .5 else 1 - aj
                #if ai > aj * (1 + kSmallEpsilon):
                  to_delete.add(snp1)
                else: 
                  to_delete.add(window[0][j])
                break
        remaining = np.array([i for i,snp in enumerate(window[0]) if snp not in to_delete])
        r2 = corrT[remaining,:][:,remaining]
        if np.max(r2**2) < threshold:
          break

      return to_delete

    if step_sz is None:
      step_sz = int(win_sz/2)
    for chrom in self.keys:
      if chrom == 'meta':
        continue 
      logging.debug("---Decentralized LD pruning on chrom: {}".format(chrom))
      # Get snps that pass the allele frequency threshold
      snps = np.sort(np.array(self.DOs[0].snps_present(chrom)).astype(int))
      win_sz = min(snps.shape[0], win_sz)
      finished, winstart  = False, 0
      highLD, to_delete = set(), set()
      while not finished:
        winend = winstart + win_sz
        if winend >= len(snps):
          finished = True 
          winend = len(snps)
        window = snps[winstart:winend] #preliminary window 
        window = np.sort(np.array(list(set(window) - to_delete)))#[:win_sz]
        to_delete = pruner(chrom, threshold, window)
        highLD = highLD.union(to_delete)
        winstart += step_sz# + offset[0][0]
      #toKeep = set(snps) - highLD
      logging.info("---Keeping {} snps after AF/LD pruning".format(len(snps) - len(highLD)))
      for DO in self.DOs:
        DO.delete_snps(chrom, highLD)

   

  def LD_prune(self,threshold, AF_threshold, win_sz, step_sz=None):
    """Flag snps that have small LD"""
    
    def pruner(chrom, threshold, window):
      window.shape = (1, window.shape[0])
      to_delete = set()
      n = window.shape[1]
      cov = np.zeros((n,n))
      # considerable optimization can be done so that only the parts 
      # that are previously not communicated get communicated 
      for DO in self.DOs:
        cov += float(DO.n)/float(self.n) * DO.give_cov([chrom], window)
      #cov /= self.nDOs
      # with covariance matrix we can be more accurate than the 
      # simple greedy we implemented in centralized but we go with the
      # same algorithm for comparison's sake 
      for i, snp in enumerate(window[0,:]):
        if snp in to_delete:
          continue
        else:
          for j in range(i+1, window.shape[1]):
            if window[0,j] in to_delete:
              continue
            elif cov[i,j]**2 > threshold:
              to_delete.add(window[0,j])
      return to_delete

    if step_sz == None:
      step_sz = int(win_sz/2)
    for chrom in self.keys:
      if chrom == 'meta':
        continue 
      logging.info("---Decentralized LD pruning on chrom: {}".format(chrom))
      # Get snps that pass the allele frequency threshold
      snps = np.sort(np.array(self.DOs[0].AF_filter(AF_threshold, chrom))).astype(int)
      win_sz = min(snps.shape[0], win_sz)
      finished, winstart  = False, 0
      highLD = set()
      i = 0
      while not finished:
        winend = winstart + win_sz
        if winend >= len(snps) - 1:
          finished = True 
          winend = len(snps) - 1
        window = snps[winstart:winend]
        window = np.sort(np.array(list(set(window) - highLD)))
        to_delete = pruner(chrom, threshold, window)
        highLD = highLD.union(to_delete)
        winstart += step_sz
        if winstart / 5000 > i:
          logging.debug("pruning at {}".format(winstart))
          i += 1
      toKeep = set(snps) - highLD
      logging.info("----Keeping {} snps after AF/LD pruning".format(len(toKeep)))
      for DO in self.DOs:
        DO.tag_snps(chrom, toKeep, 'prune_selected', True)

  def PCA(self, n_components=None, chroms=None):
    if chroms is None or chroms == []:
      chroms = [item for item in self.keys if item != 'meta']
    chroms = sorted(chroms, key=lambda x: int(x))
    DO = self.DOs[0]
    n = DO.count(list(set(self.keys) - set(chroms)))
    to_PCA = np.zeros((n, n), dtype=np.float32)
    logging.info("Preparing covariance matrix of size {}".format(n))
    for DO in self.DOs:
      DO.give_cov_pca(chroms, n, to_PCA, 1.0)# float(DO.n)/float(DO.n-1))
    if n_components is not None:
      m = min(self.n, n)
      m = min(m, n_components)
      #n_components = (n - n_components, n-1)
    #sigma, v = eig(to_PCA, overwrite_a=True, eigvals=n_components)# for linalg.eigh slow
    logging.info("Running PCA")
    sigma, v = eig(to_PCA, k=n_components, ncv=3*n_components)
    logging.info("Done running PCA")
    # there should be no ev with negativ e ev. If there is it should 
    # be tiny and due to numerical errors 

    del to_PCA
    sigma, v = zip(*sorted(zip(sigma, v.T),reverse=True))
    v = np.array(v)

    sigma = np.array(sigma)
    sigma[sigma < 0] = 0
    for DO in self.DOs:
      DO.store_eigs(sigma, v, chroms)
    #pca = PCA(n_components=n_components)
     #for now ignore the n_components arg
    #pca.fit(to_PCA)

  def change_pheno(self, pheno_plink):
    pheno_file = plinkfile.open(pheno_plink)
    sample_list = pheno_file.get_samples()
    iid = [item.iid for item in sample_list]
    status = [item.affection  for item  in sample_list]
    status_dict = dict((key, value) for (key, value) in zip(iid, status))
    for DO in self.DOs: 
      DO.update_pheno(status_dict)

  def copy_pca(self, other, local=False):
    for DO in self.DOs:
      base = os.path.basename(DO.store_name)
      file_name = os.path.join(other, base)
      DO.copy_pca(file_name, local)

  def run_regression(self, numPCs, n_iters, warm_start=True, chroms=[], sites=None, kind='ADMM',
      verbose=False, out_file="d_beta.txt"):

    def _regression(kind, verbose, **kwargs):
      """Dispatches to regression algorithm"""
      if kind == 'ADMM':
        if verbose:
          return self._ADMM_verbose(**kwargs)
        else:
          return self._ADMM(**kwargs)
      elif kind == 'AVG':
         return self._AVG(**kwargs)


    logging.info("-Running regression")
    DOs = self.DOs

    kwargs = {"rho": 10.0, "max_iters":n_iters, "alpha":1.2,
      "npcs":numPCs, "mu":0.0}#self.n * 1e-9}
    # Compute the variance of PCs
    first_moment = np.zeros((1, numPCs))
    second_moment = np.zeros((1, numPCs))
    #covp = len(pos) + numPCs
    covp = numPCs + 1

    for DO in DOs:
      DO.load_snp = True
      m1, m2 = DO.give_moments(["meta/pca_u"])
      first_moment += np.array(m1[0][:numPCs]) * DO.n / float(self.n)
      second_moment += np.array(m2[0][:numPCs]) * DO.n / float(self.n)
    stds = np.sqrt(second_moment - first_moment**2)
    kwargs["stds"] = stds

    write_n = 50
    if verbose:
      write_n = write_n / 10

    # Run for each snp
    if len(chroms) == 0 :
      chroms = self.keys
    else:
      chroms = [unicode(str(chrom)) for chrom in chroms]
    num_g = DOs[0].count(exclude=list(set(self.keys) - set(chroms)))
    pbar = tqdm.tqdm(total=num_g)
    counter, all_betas, warm_beta = 0, [], np.zeros((covp, 1))

    # Run regression with PC's only one time, to get the likelihood for the smaller model
    kwargs['pos'] = []
    kwargs["beta"] = warm_beta[1:]
    pc_beta = _regression(kind, False, **kwargs)
    pc_likelihood = 0
    
    warm_beta[1:] = pc_beta

    for DO in DOs: 
      pc_likelihood += DO.likelihood(pc_beta)
      DO.load_snp = True
      DO.current_Y = None
    
    if not verbose:
      pval = np.empty((covp + 2, 1))
    else:
      pval = np.empty((covp + 2, n_iters+1))
 
    # Run regression for everything else and compute the log likelihood difference/Wald Pvalues
    with open(out_file, 'w') as fout:
      for chrom in chroms:
        if chrom == 'meta':
          continue
        logging.info("--Running {} on chromosome: {}".format(kind, chrom))
        snps = sorted(DOs[0].dataset_keys(chrom), key=lambda x:int(x))
        pval[covp+1, :] = chrom
        for snp in snps:
          kwargs["pos"] = [(chrom, snp)]
          kwargs["beta"] = warm_beta
          beta = _regression(kind, verbose, **kwargs)
          if isnan(beta[0,0]):
            pval[:covp+1,:] = np.nan
            for DO in DOs:
              DO.load_snp = True
          else:
            likelihood = 0
            for DO in DOs:
              likelihood += DO.likelihood(beta, verbose)
            covLogit = _sum([DO.covLogit([(chrom, snp)], beta, stds, True) for DO in DOs], axis=0)
            # get pvalues
            covLogit = inverse(covLogit)
            z = (beta / sqrt(np.diag(covLogit)).reshape(covp, 1))
            z = z * z
            pval[:covp,:] = chi2sf(z, 1)
            pval[covp,:] = likelihood - pc_likelihood
          if not verbose:
            all_betas.append( "\t".join(map(str, beta[:,0])) +"\t" + "\t".join(map(str, pval[:,0]))) 
          else:
            for ind, line in enumerate(beta.T):
              all_betas.append( "\t".join(map(str, line)) +"\t" + "\t".join(map(str, pval[:,ind].tolist() + [ind])))


          counter += 1
          if counter == write_n:
            fout.write('\n'.join(all_betas))
            fout.write('\n')
            counter = 0
            all_betas = []
            pbar.update(write_n)
      fout.write('\n'.join(all_betas))

  def _ADMM(self, pos, npcs, rho, beta, alpha=1., max_iters=10, mu=0.0, stds=1, #1e-9, stds = 1,
      logistic=True, verbose=True): # mu is really self.n * mu
    """Performs ADMM regression. So far, only logistic regression is implemented."""
    DOs = self.DOs
    covp = len(pos) + npcs
    K = len(DOs)
    z = np.zeros((covp, K))
    u = np.zeros((covp, K))
#    shrink_param = mu / float(rho * K)
    for k in xrange(max_iters):
      for i, DO in enumerate(DOs): # can be parallelized 
        try:
          # z update:
          z[:,i] = DO.admm_update(pos, npcs,u[:,i, None], beta, rho, z[:,i, None], stds, logistic, covp)
        except ValueError:
          beta *= np.nan
          return beta
      # Update betas
      z_hat = add(alpha * z, sub(1.0, alpha) * beta)
#      meanVal = div(_sum(add(z_hat, u), 1)[:,None], K)
#      beta = div(_sum(add(z_hat, u), 1)[:,None], K)
      beta = div(umr_sum(z_hat,1)[:,None], K)
#      beta = sub(maximum(0, sub(meanVal, shrink_param)), maximum(0, -add(meanVal, shrink_param)))
      # Update u:
      u += sub(z_hat, beta)
    return beta


  def _ADMM_verbose(self, pos, npcs, rho, beta, alpha=1.0, max_iters=10, mu=1e-9, stds=1, 
      logistic=True):
    """Same as _ADMM except records the beta after every iteration. _ADMM avoids checking the 
    condition over and over again. Probably a stupid optimization but w/e"""
    DOs = self.DOs
    covp = len(pos) + npcs
    K = len(DOs)
    z = np.zeros((covp, K))
    u = np.zeros((covp, K))
    shrink_param = mu / float(rho * K)
    Betas = np.empty((covp, max_iters+1))
    Betas[:,0] = 0
    Us = np.empty((1, max_iters+1))
    Us[0,0] = 0
    for k in xrange(max_iters):
      for i, DO in enumerate(DOs): # can be parallelized 
        try:
          # z update:
          z[:,i] = DO.admm_update(pos, npcs,u[:,i, None], beta, rho, z[:,i, None], stds, logistic, covp)
        except ValueError:
          Betas[k+1:, :] = np.nan
          return beta
      # Update betas
      z_hat = add(alpha * z, sub(1.0, alpha) * beta)
      #meanVal = div(_sum(add(z_hat, u), 1)[:,None], K)
      #beta = sub(maximum(0, sub(meanVal, shrink_param)), maximum(0, -add(meanVal, shrink_param)))
      beta = div(umr_sum(add(z_hat, u), 1)[:,None], K)
      Betas[:,k+1] = beta[:,0]
      # Update u:
      u += sub(z_hat, beta)
      Us[0,k+1] = np.linalg.norm(u)
    return Betas

  def _AVG(self, pos, npcs, stds = 1, logistic=True, verbose=True, **kwargs): 
    """Performs Average regression. So far, only logistic regression is implemented.
    Performs the regression on de-centralized data. This simply averages all the results,
    for the actual analysis, we used inverse variance weighted averaging FE model."""
    covp = len(pos) + npcs
    DOs = self.DOs
    N = float(self.n)

    beta = np.zeros((covp, 1))
    for i, DO in enumerate(DOs): # can be parallelized 
#      try:
        beta += DO.run_regression(pos, npcs, beta, stds, logistic, covp).T * DO.n / N
#      except ValueError:
#        beta *= np.nan
#        return beta
    # Update betas
    return beta



  def PCA_Centralized(self, n_components=None, chroms=None):
    from sklearn.decomposition import PCA 
    if chroms is None or chroms == []:
      chroms = [item for item in self.keys if item != 'meta']
    chroms = sorted(chroms, key=lambda x: int(x))
    DO = self.DOs[0]
    n = DO.count(list(set(self.keys) - set(chroms)))
    data = np.empty((self.n, n), dtype=np.float32)
    logging.info("centralizing data just to run centralized PCA")
    start = 0
    for DO  in self.DOs: 
      data[start:start+DO.n,:] = DO.give_data(chroms,n)
      start += DO.n

    pca = decomp.PCA()
    U, S, V = pca._fit_truncated(data, n_components=n_components, svd_solver = 'arpack')
#    u, sigma, vt = randomized_svd(data, n_components, transpose=False)
#    u,vt = svd_flip(u, vt, u_based_decision=False)
    self.DOs[0].record_centralized_pca(S, U) 
    logging.info("Done with centralized PCA")

  def run_meta_filters(self, t_missing=None, t_AF=None, t_hwe=None, t_LD=None, win_sz=50, global_clean=False):
    def count(global_clean):
      unfiltered = 0
      for chrom in self.keys:
        if chrom == 'meta':
          continue
        present = self.DOs[0].locally_unfiltered(chrom)
        for DO in self.DOs[1:]:
          present = present.intersection(DO.locally_unfiltered(chrom))
        unfiltered += len(present)
        if global_clean: 
          for DO in self.DOs:
            DO.clean_by_local_filter(chrom, present)
      return(unfiltered)
    if t_missing is not None:
      logging.info("Starting local missing filter")
      for DO in self.DOs: 
        DO.local_missing_filter(t_missing)
      unfiltered = count(global_clean)
      logging.info("After missing rate filter {} snps remain".format(unfiltered))
    if t_AF is not None:
      logging.info("Starting local AF")
      for DO in self.DOs:
        DO.local_AF_filter(t_AF)
      unfiltered = count(global_clean)    
      logging.info("After AF filter {} snps remain".format(unfiltered))
    if t_hwe is not None:
      logging.info("Starting HWE filter")
      for DO in self.DOs:
        DO.local_HWE_filter(t_hwe)
      unfiltered = count(global_clean)
      logging.info("After HWE filter {} snps remain".format(unfiltered))
    if t_LD is not None:
      logging.info("Running LD filter")
      for DO in self.DOs:
        DO.local_LD_filter(t_LD, win_sz) #implement
      unfiltered = count(global_clean)
      logging.info("After LD filter {} snps remain".format(unfiltered))

  def run_local_pca(self, n_components=10, chroms=None):
    for DO in self.DOs: 
      DO.local_pca(n_components, chroms)

  def run_meta_regression(self, numPCs, out_file):
    logging.info("Starting meta regression...")
    chroms = self.keys
    with open(out_file, 'a') as fout:
      for chrom in chroms:
        if chrom == 'meta': 
          continue
        logging.info("Moving on to chrom " + chrom)
        for i, DO in enumerate(self.DOs):
          betas, standard_errors, pvals = DO.local_regression(numPCs, chrom)
          if not i: # first DO
            to_write = np.empty((len(betas), 3*len(self.DOs)+1))
          to_write[:,i] = betas[:,0]
          to_write[:,i+len(self.DOs)] = standard_errors[:,0]
          to_write[:,i+2*len(self.DOs)] = pvals[:,0]
        to_write[:,3*len(self.DOs)] = chrom
        np.savetxt(fout, to_write)
    logging.info("Finished Meta-regressions")




  def impute (self):
    for DO in self.DOs: 
      DO.impute()
      logging.info("DUUUUDE")

class add_dict(dict):
  def set_key_values(self, keys=None, value=None):
    if keys is None:
      keys = self.keys()
    if value is None:
      value = 0
    for key in keys:
      self[key] = value


  def update(self, other, frac=1.0, pos=None):
    if pos is None:
      k1 = other.keys()[0]
      if isinstance(other[k1], int):
        for key, value in other.iteritems():
          dicVal = self[key]
          self[key] = dicVal + frac * value
      else:# it is an array
        for key, value in other.iteritems():
          dicVal = self[key]
          self[key] = [x + frac * y for x,y in zip(dicVal, value)]
    elif pos == 0: #deal with these later TODO they can be put in the framework above
      for key, value in other.iteritems():
        dicVal = self[key]
        self[key] = dicVal[0] + value[2] * value[0], dicVal[1], dicVal[2] + value[2]
    elif pos == 1:
      for key, value in other.iteritems():
        dicVal = self[key]
        self[key] = dicVal[0] + value[0]**2, dicVal[1] + value[1]

if __name__=='__main__':
  print "no commands here yet. Test using WTCCC_run.py"


