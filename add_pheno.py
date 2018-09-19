from plinkio import plinkfile
import numpy as np
import os, tqdm, time, sys
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import sklearn.decomposition as decomp
from sklearn.metrics import log_loss
from scipy.stats import chi2
from memory_profiler import profile
import glob
#import logging 
SCRATCH = "/local/scratch/armin"

def setup_custom_logger(name):
  formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
  handler = logging.FileHandler('Centralized_GWAS.log', mode='w')
  handler.setFormatter(formatter)
  screen_handler = logging.StreamHandler(stream=sys.stdout)
  screen_handler.setFormatter(formatter)
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.addHandler(handler)
  logger.addHandler(screen_handler)
  return logger

logger = setup_custom_logger('add_pheno')

def add_pheno(plink_in, multigenecity, out, h=0.85, p_cases=0.5):
  plink_file = plinkfile.open( plink_in )
  if not plink_file.one_locus_per_row( ):
    print( "This script requires that snps are rows and samples columns." )
    exit(1)
  
  sample_list = plink_file.get_samples()
  
  locus_list = plink_file.get_loci()
  n = len(sample_list)
  p = len(locus_list)
  edge_offset = 100
  causal_mut_index = np.linspace(edge_offset, p-edge_offset, multigenecity, dtype=int)
  gen_effect_size_unnormalized = {item: np.random.normal(loc=0, 
    scale=float(h)/np.sqrt(multigenecity)) for item in causal_mut_index}
  print causal_mut_index
  causal_mutations = set()
  mutation_meta = {}
  prs = np.zeros(n)
  for i, variant in enumerate(locus_list): 
    row = plink_file.next()
    if i in causal_mut_index:
      genotypes = np.fromiter(row, dtype=float)
      genotypes[genotypes==3] = np.mean(genotypes[genotypes!=3])
      prs += genotypes * gen_effect_size_unnormalized[i]

  plink_file.close()
  del causal_mut_index, gen_effect_size_unnormalized
  env_rs_unnormalized = np.random.normal(loc=0, scale=np.sqrt(1-h**2), size=n)

  gen_effect_size = h * (prs - np.mean(prs)) / np.std(prs)
  env_effect_size = np.sqrt(1-h**2) * (env_rs_unnormalized - np.mean(env_rs_unnormalized)
      ) / np.std(env_rs_unnormalized)

  burden = gen_effect_size + env_effect_size
  sorted_i = np.argsort(burden)[::-1]
  
  ncases = int(n * p_cases)
  cases_i = set(sorted_i[:ncases])
  
  # write new plink file 
  for i, sample in enumerate(sample_list):
    sample_list[i].affection = int(i in cases_i)

  #plink_write = plinkfile.create(out, sample_list)
  plink_write = plinkfile.WritablePlinkFile( out, sample_list )
  #plinkio doesn't have seek? so we close it when we don't need it and reopen it here
  plink_file = plinkfile.open( plink_in )
  for i, variant in tqdm.tqdm(enumerate(locus_list)): 
    row = plink_file.next()
    plink_write.write_row(variant, row)
  
  plink_write.close()
  plink_file.close()


def run_gwas(imputed, toPCA, out, npcs=5):
  # Compute PCA
  plinkpca = plinkfile.open(toPCA)
  if not plinkpca.one_locus_per_row():
    print("The plink file is fucked")
    exit(1)

  sample_list = plinkpca.get_samples()
  locus_list = plinkpca.get_loci()

  demo = pd.read_table('data/popres_European.ind', delimiter='\t')
  famIDs = set(int(i.fid) for i in sample_list)
  demography = [row.country for _, row in demo.iterrows() if row.famID in  famIDs]
  demo = pd.read_table('data/popres_European.ind', delimiter='\t')
  ids = [int(row.famID) for _, row in demo.iterrows() if row.famID in famIDs]

  del demo, famIDs

  n = len(sample_list)
  p = len(locus_list)
  gen_mat = np.empty((n, p), dtype=np.float32)
  
  loc = 0
  for i, row in enumerate(plinkpca):
    arr = np.fromiter(row, dtype=np.float32)
    arr[arr==3] = np.nan
    sd = np.nanstd(arr)
    mu = np.nanmean(arr)
    arr -= mu 
    arr[np.isnan(arr)] = 0
    arr /= sd
    gen_mat[:,loc] = arr#np.fromiter(row, dtype=np.float32)
    loc += 1

  pca = decomp.PCA()
  U, S, V = pca._fit_truncated(gen_mat, n_components=npcs, svd_solver='arpack')
  
  np.savetxt(out+'.V.txt',V)
  np.savetxt(out+'.U.txt',U)
  np.savetxt(out+'.sigma.txt',S)
  np.savetxt(out+'.ids.txt', ids, fmt='%i')
  with open(out+'.countries', 'w') as f:
    f.write("\n".join(demography))
  
  #del S, V, gen_mat
  U_id_dict = dict((key, value) for (key, value) in zip(ids, U[:,:npcs]))
  run_regressions(imputed, U_id_dict, out+'betas.txt', npcs)
#  np.save(out+'betas.txt' , pvals)

def run_regressions(plink_file, U_dict, out_file, npcs, buf=100 ):
  plink_file = plinkfile.open(plink_file)
  if not plink_file.one_locus_per_row():
    print("The plink file is fucked")
    exit(1)

  locus_list = plink_file.get_loci()
  sample_list = plink_file.get_samples()
  n = len(sample_list)
  p = len(locus_list)
  y = np.array([sample.affection for sample in sample_list])


  X = np.empty((n, npcs + 1))
  betas = np.empty((buf, 2 * X.shape[1] + 2), dtype = np.float32)
  #X_design = np.ones((n,2))
  V = np.matrix(np.zeros(shape = (X.shape[0], X.shape[0])))
  X[:,1:] = [U_dict[int(sample.iid)] for sample in sample_list]
  X[:,1:] /= np.std(X[:,1:], axis = 0)
  covp = X.shape[1]
  
  # High C corresponds to less regularization. 
  model = LogisticRegression(fit_intercept=False,  tol=1e-5, C=1e4)
  k = 0
  # fit nuisance
  model.fit(X[:,1:], y)
  y_model = model.predict_proba(X[:,1:])
  l_null = log_loss(y, y_model, normalize=False)

  with open(out_file, 'w') as out_f:
    i = 0
    logging.info("Iterating over SNPs")
    for j, row in tqdm.tqdm(enumerate(plink_file), total = p):
      locus = locus_list[j]
      arr = np.fromiter(row, dtype = np.float32)
      mu = np.mean(arr[arr!=3])
      std = np.std(arr[arr!=3])
      arr[arr==3] = mu
      arr -= mu
      if std > 0:
        arr /= std
        X[:,0] = arr
        model.fit(X, y)
        # Wald Test
        y_model = model.predict_proba(X)
        X_design= X
        np.fill_diagonal(V, np.multiply(y_model[:,0], y_model[:,1]))
        covLogit = np.linalg.inv(X_design.T * V * X_design)
        coefs = np.array(model.coef_)#np.insert(model.coef_, 0, model.intercept_)
        z = (coefs / np.sqrt(np.diag(covLogit))) ** 2
        # Chi-squared test 
        l_fit = log_loss(y, y_model, normalize=False)
        D = l_fit - l_null
        p = chi2.sf(z, 1)
        betas[i, :covp] = coefs
        betas[i, covp:2*covp] = p
        betas[i, 2*covp] = D
      else: 
        betas[i,:] = np.nan
      betas[i, 2*covp+1] = locus.chromosome
      i += 1
      if i == buf:
        i = 0
        np.savetxt(out_f, betas, delimiter='\t')
    np.savetxt(out_f, betas[:i,:], delimiter='\t')  # write the remaining
    logging.info("Finished iterating")

if __name__=='__main__':
  from shutil import copyfile, copy
  basename = "popres_Europeangeno05maf05"
  sim_out = "simulated"
  outname = 'centralized.txt'
  if not os.path.exists(SCRATCH):
    os.makedirs(SCRATCH)

  scratch_path_fam = os.path.join(SCRATCH, basename+".fam")
  scratch_path_bed = os.path.join(SCRATCH, basename+".bed")
  scratch_path_bim = os.path.join(SCRATCH, basename+".bim")
  
  copyfile(os.path.join('data', basename+".fam"), scratch_path_fam)
  copyfile(os.path.join('data', basename+".bed"), scratch_path_bed)
  copyfile(os.path.join('data', basename+".bim"), scratch_path_bim)

  np.random.seed(123)

  logger.info("Adding phenotypes")
  add_pheno(os.path.join(SCRATCH, basename), 10, sim_out)
  logger.info("Phenotype added")
  os.remove(scratch_path_fam)
  os.remove(scratch_path_bed)
  os.remove(scratch_path_bim)

  np.random.seed(456)

  sim_scratch_path = os.path.join(SCRATCH, sim_out)
  pca_file_path = os.path.join('data', "popres_Europeangeno05maf05hwe10indppair502502")

  copyfile(sim_out+".bed", sim_scratch_path+".bed")
  copyfile(sim_out+".bim", sim_scratch_path+".bim")
  copyfile(sim_out+".fam", sim_scratch_path+".fam")
  sim_out = os.path.join(SCRATCH, outname)
  logger.info("Running Centralized GWAS")
  run_gwas(sim_scratch_path, pca_file_path, sim_out)
  logger.info("Exiting")
  # Copy all generated files back 
  for filename in glob.glob(sim_out+'*'):
    copy(filename, ".")

