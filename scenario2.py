# Coder: Armin Pourshafeie
# runs a small decentralized GWAS. There are two centers, one containing the cases,
# the other containing the controls

import os, sys
import argparse
import process_chiamo
import analytics
import logging
import datetime
import pdb
import h5py
import numpy as np
from functools import partial
from qc_utils import *
from scenario1 import *
from plinkio import plinkfile
from shutil import *
import glob

SCRATCH = "/local/scratch/armin/case_control_split/"

def case_control_split(to_split, num_case_holders, num_control_holders, split_prefix, seed=1234, pheno_file=None, create=True):
  """Distributes the rows of h5py dataset at to_split into num_case_holders, num_control_holders groups 
  of approximately equal size adding up to the total number of individuals
  This function copies by shamelessly iterating over everything so it can be 
  very slow"""
  # Figure out how many #cases and #controls
  if pheno_file is None:
    with h5py.File(to_split, 'r') as to_split_fp:
      status = to_split_fp['meta/Status'].value
      num_cases = np.sum(status)
      controls = status==0
      num_controls = np.sum(controls)
      control_rows = np.where(controls)
      case_rows    = np.where(~controls)
    del controls
  else: # It must be a plink file
    plink_file = plinkfile.open(pheno_file)
    sample_list = plink_file.get_samples()
    status = np.array([i.affection for i in sample_list])
    ids    = np.array([i.iid for i in sample_list])
    case_rows = ids[status==1]
    control_rows   = ids[status == 0]
    num_cases    = len(case_rows)
    num_controls = len(control_rows)
    del status, ids

  if num_case_holders > 1:
    case_per_silo = [num_cases/num_case_holders] * (num_case_holders -1)
    case_per_silo.append(num_cases - sum(case_per_silo))
  else:
    case_per_silo = [num_cases]
  to_create = zip(case_per_silo, ['case'] * num_case_holders)
  
  if num_control_holders > 1:
    control_per_silo = [num_controls/num_control_holders] * (num_control_holders -1)
    control_per_silo.append(num_controls - sum(control_per_silo))
  else: 
    control_per_silo = [num_controls]

  to_create += zip(control_per_silo, ['control'] * num_control_holders)
  to_create = set(to_create)

  names = []
  def group_copy(name, node, rows, fp):
    dtype = node.dtype
    value = node[...]
    fp.require_dataset(name, data=value[rows], shape=(len(rows),), dtype=dtype)


  i = 0
  with h5py.File(to_split, 'r') as to_split_fp:
    if pheno_file is not None:
      ids = to_split_fp["meta/id"].value
      case_rows = np.where([ind in case_rows for ind in ids])[0]
      control_rows = np.where([ind in control_rows for ind in ids])[0]

    np.random.seed(seed)
    case_rows = np.random.permutation(case_rows)
    control_rows = np.random.permutation(control_rows)
    while len(to_create):
      count, status = to_create.pop()
      split_name = split_prefix + status + str(i) + '.h5py'
      names.append(split_name)
      if not create: 
        i += 1
        continue
      logging.info("-Constructing: " + split_name)
      if status == 'case':
        chosen_rows = case_rows[:count]
        case_rows   = case_rows[count:]
      else:
        chosen_rows  = control_rows[:count]
        control_rows = control_rows[count:]

      with h5py.File(split_name, 'w') as copy_to_fp: 
        for key in to_split_fp.keys():
          dset_to_copy = to_split_fp[key]
          dset_to_copyto = copy_to_fp.require_group(key)
          copier = partial(group_copy, rows=chosen_rows, fp=dset_to_copyto)
          dset_to_copy.visititems(copier)
      i += 1

  return names


def mv_from_to_scratch(fname, to=True, delete=False, pattern=""):
  """copies the file to scratch"""
  if len(pattern)>0:
    if to:
      src = fname
    else:
      src = os.path.join(SCRATCH, fname)
    for filename in glob.glob(src + pattern):
      loc = mv_from_to_scratch(filename, to, delete, "")
      dirname = os.path.dirname(loc)
      base    = os.path.basename(fname)
    return os.path.join(dirname, base)

  loc = ""
  if to:
    src = fname
    dest = os.path.join(SCRATCH, src)
    loc = dest
  else:
    dest = fname
    src = os.path.join(SCRATCH, dest)
    loc = src
  destdir = os.path.dirname(dest)
  if len(destdir) > 0  and not os.path.exists(destdir):
    os.makedirs(destdir)
  copy(src, dest)
  if not to and delete:
    os.remove(src)
  return loc
    
#def backup(src, dest, ext=[]):
#  if len(ext) == 0:
#    if os.path.isdir(src):
#      src = os.path.normpath(src)
#      dirname = src.split(os.sep)[-1]
#      if os.path.isdir(os.path.join(dest, dirname)):
#        rmtree(os.path.join(dest, dirname))
#      copytree(src, os.path.join(dest, dirname))
#    else:
#      copy(src, dest)
#  else:
#    if os.path.isdir(src):
#      dirname = src.split(os.sep)[-1]
#      newdir = os.path.join(dest, dirname)
#      if os.path.isdir(newdir):
#        rmtree(newdir)
#      os.mkdir(newdir)
#      for filename in os.listdir(src):
#        extension = os.path.splitext(filename)[1]
#        if extension in ext:
#          copy(os.path.join(src, filename), os.path.join(newdir, filename))
#    else:
#      for extension in ext:
#        copy(src+extension, dest+extension)
#



def run_experiment(shuffled=True):
  plink = "/srv/gsfs0/software/plink/1.90/plink"
  plink_dem = 'data/popres_European.ind'
  plink_file = 'data/popres_European'
  store_name = 'POPRES_NoSpain.h5'
  pheno_file = 'simulated'
  split_data_dir = os.path.join(SCRATCH, "case_control/")
  cwd = os.getcwd()
  args = parse_arguments()
  
  logname = os.path.join(args.log_dir, "separate.log")
  print("Log locate at {}".format(logname))
  setup_logger('decentralized', os.path.join(args.log_dir, 'sep.log'))
 

############################## 
  if not os.path.exists(SCRATCH):
    os.makedirs(SCRATCH)

  sstore_name = os.path.join(SCRATCH, store_name)
  if not os.path.isfile(store_name):
    backup(plink_file, os.path.join(SCRATCH, plink_file), [".bed", ".bim", ".fam", ".ind"])
    splink_file = os.path.join(SCRATCH, plink_file)
    splink_dem = os.path.join(SCRATCH, plink_dem)
    process_chiamo.read_plink(splink_file, splink_dem, shuffled, store_name=sstore_name)
    backup(os.path.join(SCRATCH, store_name), store_name)
  else:
    logging.info("-SNP file has already been made")
    if not os.path.isfile(sstore_name):
      backup(store_name, os.path.join(SCRATCH, store_name))

  # Copy a version for future use 
  if not os.path.isfile(store_name[:-3]+"copied.h5"):
    copyfile(sstore_name, store_name[:-3]+"copied.h5")
  store_name = sstore_name
  centralized_DO = analytics.DO(store_name=store_name)
  logging.info("File contains {} snps".format(centralized_DO.count()))

  # Split the dataset into parts for decentralized analysis
  if not os.path.isdir(split_data_dir):
    os.mkdir(split_data_dir)

####################################### QC pipeline
#
#  backup(pheno_file, os.path.join(split_data_dir, pheno_file), [".bed", ".bim", ".fam"])
#  backup('data', SCRATCH)
#  os.chdir(split_data_dir)
#  names = case_control_split(store_name, 1, 1, split_data_dir, pheno_file=pheno_file, create=True)
#  hub = analytics.Center(names)
#  hub.change_pheno(pheno_file)
#  backup(split_data_dir, "_backup/", ext=['.h5py', ".h5"])
#  hub.loci_missing_rate_filter(0.05)
#  backup(split_data_dir, "_backup_missing/", ext=['.h5py', ".h5"])
#  qc_compare(names[0], plink + " --bfile ../data/popres_European --not-chr 23 25 --geno 0.05 --make-bed", "geno05")
#  hub.MAF_filter(0.05)
#  backup(split_data_dir, "_backup_MAF/", ext=['.h5py', ".h5"])
#  qc_compare(names[0], plink + " --bfile geno05 --maf 0.05 --make-bed",  "geno05maf05")
#  hub.HWE_filter(1e-10)
#  backup(split_data_dir, "_backup_HWE/", ext=['.h5py', ".h5"])
#  qc_compare(names[0], plink + " --bfile {}  --hwe 1e-10 --make-bed".format( "geno05maf05")
#      ,  "geno05maf05hwe10")
#  hub.correct_LD_prune(0.2, 50)
#  runcmd(plink + " --bfile {} --indep-pairwise 50 25 .2 --make-bed --out {} ".format( "geno05maf05hwe10", "intermediate"))
#  qc_compare(names[0], plink + " --bfile geno05maf05hwe10 --exclude intermediate.prune.out  --make-bed ", "geno05maf05hwe10indppair502505")
#  backup(split_data_dir, "_backup_pruned/", ext=['.h5py', ".h5"])
#  hub.normalize()
#
#  backup(SCRATCH, os.path.join(cwd, "case_control_analysis"))
#
#
##################################### Meta study analysis
##  # FIlter based on local_missing, local_MAF, local_LD , do local PCA
##  # Transform, run local regression, average results 
##
#
#  base = '_backup_missing/' # Meta will always be more conservative
#  metaFiles = "meta_analysis/"
#  backup(base, metaFiles, ext=['.h5py', ".h5"])
#
#  hub_names = [os.path.join(metaFiles, name) for name in os.listdir(metaFiles)]
#  hub = analytics.Center(hub_names)
#  hub.run_meta_filters(t_missing=0.05, t_AF=None, t_hwe=None, t_LD=None, global_clean=True)
#  logging.info(compare(hub_names[0], "geno05"))
#  base = '_backup_MAF/' # Meta will always be more conservative
#  backup(base, metaFiles, ext=['.h5py', ".h5"])
#  hub.run_meta_filters(t_missing=0.05, t_AF=0.05, t_hwe=None, t_LD=None, global_clean=True)
#  logging.info(compare(hub_names[0], "geno05maf05"))
#  hub.run_meta_filters(t_hwe=1e-10, global_clean=True)
#  logging.info(compare(hub_names[0], "geno05maf05hwe10"))
#  hub.run_meta_filters(t_missing=None, t_AF=None, t_hwe=None, t_LD=.2, global_clean=True)
#  logging.info(compare(hub_names[0], "geno05maf05hwe10indppair502505"))
#
#  backup(SCRATCH, os.path.join(cwd, "case_control_analysis"))


###################################### PCA 
#  # Stopped here to get a high mem machine
#  if  os.path.exists(SCRATCH):
#    rmtree(SCRATCH)
#
#  backup('case_control_analysis/case_control_split/', '/local/scratch/armin')
#  os.chdir(split_data_dir)
#  names = case_control_split(store_name, 1, 1, split_data_dir, pheno_file=pheno_file, create=False)
#  hub = analytics.Center(names)
#  chroms =  None #['1', '2', '3', '4', '5', '6', '7', '8', '9']# none#  hub.PCA(chroms=chroms, n_components=5)
#  hub.PCA(chroms=chroms, n_components=5)
#  backup(split_data_dir, "_backup_PCA/", ext=[".h5py", ".h5"])
#  backup(SCRATCH, os.path.join(cwd, "case_control_analysis"))
#  rmtree(SCRATCH)
#
#
##################################### Regression
#
#  Dbeta = "CC_D_betas.txt"
#  if  os.path.exists(SCRATCH):
#    rmtree(SCRATCH)
#
#  backup('case_control_analysis/case_control_split/', '/local/scratch/armin')
#  os.chdir(split_data_dir)
#### PUll in the entire data since we deleted stuff 
#  base = '_backup_MAF'
#  regression_dir = '_regression/'
#  copy_files(base, regression_dir)
#  data_names = []
#  for fname in os.listdir(regression_dir):
#    data_names.append(os.path.join(regression_dir, fname))
#
#  full_hub = analytics.Center(data_names)
#  full_hub.copy_pca(split_data_dir)
#  full_hub.normalize()
#
#  backup(SCRATCH, os.path.join(cwd, "case_control_analysis"))
####  full_hub.impute()  # Forget about this for now 
#
#  full_hub.run_regression(5,100, out_file=Dbeta)
#  logging.info("GWAS finished!")
#  backup(Dbeta, os.path.join(cwd, Dbeta))

  iterName = "CC_17_betas_iters.txt"
# Check timing
  full_hub.run_regression(5, 100, chroms=[17], verbose=True, out_file=iterName)
  logging.info("Finished timing experiment")
  backup(iterName, os.path.join(cwd, iterName))


if __name__=='__main__':
  run_experiment()

