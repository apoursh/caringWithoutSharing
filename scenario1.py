# Author: Armin Pourshafeie
# Randomly distributes the cohort into 5 different silos and runs a decentralized GWAS.
# This script runs both our pipeline and a meta-study pipeline based on inverse-
# variance weighting (FE pipeline)
# Here Decentralized only means that the data is in different locations and does 
# not get copied into a centralized locationa.


import os, sys
import argparse
import process_chiamo
import analytics
import logging
import datetime
import h5py
import numpy as np
from functools import partial
from qc_utils import *
import pdb
from shutil import *
import glob

SCRATCH = "/local/scratch/armin/five_way_split/"

def parse_arguments():
  parser = argparse.ArgumentParser(description="""
    Run a comparison between centralized and decentralized GWAS""")
  parser.add_argument('--data_to_use', dest='data_address',
      default='Dsets_to_include.txt')
  parser.add_argument('--log_dir', dest='log_dir', default='logs/')
  
  args = parser.parse_args([])
  return args

# Unused
def exclude_snps(store_name,ind_file):
  logging.info("Filtering bad snps")
  old_chrom = None
  exclude_snp = set()
  exclude_snp_rsid = set()
  exclude_pos = set()
  def delete_in_set(name,node):
    if node.attrs['rsid'] in exclude_snp_rsid or node.attrs['snp'] in exclude_snp:
      exclude_pos.add(name)

  with h5py.File(store_name,'a') as store, open(ind_file, 'r') as snp_fp:
    for line in snp_fp: 
      if line[0] == '#': # Skip header 
        continue
      line = line[:-1] 
      chrom, affy_id, rsid, filter_name = line.split('\t')
      if len(chrom) == 1:
        chrom = '0' + chrom

      if chrom != old_chrom and old_chrom is not None: 
        logging.debug("--Now filtering bad snps in chrom {}.".format(old_chrom))
        dset = store[old_chrom]
        dset.visititems(delete_in_set)
        for name in exclude_pos:
          del dset[name]
        exclude_snp = set()
        exclude_pos = set()
        exclude_snp_rsid = set()
      exclude_snp_rsid.add(rsid)
      exclude_snp.add(affy_id)
      old_chrom = chrom


#Unused
def exclude_individuals(store_name, file_name):
  """Removes individuals. This is very slow, it figures out what rows need to be deleted,
  stores the other rows, deletes the entire dataset and finally  """
  #with h5py.File(store_name, 'a') as store, with open(file_name, 'r') as fp:
  logging.info("Filtering individuals By file: {}".format(file_name))
  inds_to_exclude = set()
  
  def filter_dataset(name, node, to_keep, store):
    if isinstance(node, h5py.Dataset):
      if node.shape != to_keep.shape:  # Already changed 
        return
      store.require_dataset(name+"tmp", data=node.value[to_keep],
          shape=(np.sum(to_keep),), dtype=node.dtype)
      del store[name]
      store[name] = store[name+"tmp"]
      del store[name+"tmp"]

  with open(file_name, 'r') as inds_fp:
    for line in inds_fp:
      if line[0] != '#': #ignore comments 
        _, _, ind, filt = line.split('\t')
        inds_to_exclude.add(ind)
  # findout in what positions these individuals appear
  with h5py.File(store_name, 'a') as store:
    # Learn the order of individuals
    wtids = store['meta/id']
    to_keep = np.ones((wtids.shape[0],), dtype=bool)
    for i, wtid in enumerate(wtids):
      if wtid in inds_to_exclude:
        to_keep[i] = False
    # read each dataset, store people to keep, delete dataset and rewrite it :( 
    # well this isn't working (not sure why) but looks like I'll have to
    # loop over stuff like a peasant...
#    copier = partial(filter_dataset, to_keep=to_keep, store=store)
#    store.visititems(copier)
    for group in store:
      logging.debug("Filtering individuals from chromsome {}".format(group))
      grp = store[group]
      for dset in grp:
        values = grp[dset].value[to_keep]
        del grp[dset]
        grp.require_dataset(dset, data=values, shape=values.shape, dtype=values.dtype)




def setup_logger(name, logname):
  #formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", 
  #    datefmt="%Y-%m-%d %H:%M:%S")
  formatter = "%(asctime)s %(levelname)-8s %(message)s"
  logging.basicConfig(filename=logname, filemode='a', level=logging.DEBUG, format=formatter)
  logger = logging.getLogger(name)
  handler = logging.FileHandler(logname, mode='a')
  handler.setFormatter(formatter)
  screen_handler = logging.StreamHandler(stream=sys.stdout)
  screen_handler.setFormatter(formatter)
  screen_handler.setLevel(logging.DEBUG)
  logger.setLevel(logging.DEBUG)
  logger.addHandler(handler)
  logger.addHandler(screen_handler)

def backup(src, dest, ext=[]):
  """utility for copying files and folders"""
  if len(ext) == 0:
    if os.path.isdir(src):
      src = os.path.normpath(src)
      dirname = src.split(os.sep)[-1]
      if os.path.isdir(os.path.join(dest, dirname)):
        rmtree(os.path.join(dest, dirname))
      copytree(src, os.path.join(dest, dirname))
    else:
      copy(src, dest)
  else:
    if os.path.isdir(src):
      dirname = src.split(os.sep)[-1]
      newdir = os.path.join(dest, dirname)
      if os.path.isdir(newdir):
        rmtree(newdir)
      os.mkdir(newdir)
      for filename in os.listdir(src):
        extension = os.path.splitext(filename)[1]
        if extension in ext:
          copy(os.path.join(src, filename), os.path.join(newdir, filename))
    else:
      for extension in ext:
        copy(src+extension, dest+extension)
  


def dset_split(to_split, num_splits, n_tot, split_prefix, seed=12345, create=True):
  """Distributes the rows of an h5py dataset at to_split into num_splits groups 
  of similar size
  This function copies by shamelessly iterating over everything so it can be 
   slow"""
  np.random.seed(seed)
  while (True):
    num = np.random.poisson(n_tot / float(num_splits), num_splits - 1)
    num = np.append(num, n_tot - np.sum(num))
    if all(num > 0):
      break

  names = []
  def group_copy(name, node, rows, fp):
    dtype = node.dtype
    value = node[...]
    fp.require_dataset(name, data=value[rows], shape=(len(rows),), dtype=dtype)
  perms = np.random.permutation(n_tot)
  current_count = 0
  with h5py.File(to_split, 'r') as to_split_fp:
    for i, number in enumerate(num):
      split_name = split_prefix + str(i) + '.h5py'
      names.append(split_name)
      logging.info("-Constructing: " + split_name)
      chosen_rows = perms[current_count:current_count+number]
      current_count += number
      with h5py.File(split_name, 'w') as copy_to_fp: 
        for key in to_split_fp.keys():
          dset_to_copy = to_split_fp[key]
          dset_to_copyto = copy_to_fp.require_group(key)
          if key != 'meta':
            copier = partial(group_copy, rows=chosen_rows, fp=dset_to_copyto)
            dset_to_copy.visititems(copier)
          else:
            group_copy("Status", dset_to_copy['Status'], chosen_rows,
                dset_to_copyto)
            group_copy("regions", dset_to_copy['regions'], chosen_rows,
                dset_to_copyto)
            group_copy("id", dset_to_copy['id'], chosen_rows, dset_to_copyto)


  return names




def run_experiment(shuffled=True):
  plink = "/srv/gsfs0/software/plink/1.90/plink"
  plink_dem = 'data/popres_European.ind'
  plink_file = 'data/popres_European'
  store_name = 'POPRES_NoSpain.h5'
  pheno_file = "simulated" 
  global_loc = "uniformSplit/"
  split_data_dir = os.path.join(SCRATCH, "five_way/")
  args = parse_arguments()
  cwd = os.getcwd()
  
  logname = "five_way.log"
  logloc = os.path.join(args.log_dir, logname)
  print("Log file is at {}".format(logloc))
  setup_logger('five_way', logloc)

#####################################  ##
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
    logging.info("-SNP file has already been created")
    if not os.path.isfile(sstore_name):
      backup(store_name, os.path.join(SCRATCH, store_name))

  if not os.path.isfile(store_name[:-3]+"copied.h5"):
    copyfile(sstore_name, store_name[:-3]+"copied.h5")
  store_name = sstore_name 
  centralized_DO= analytics.DO(store_name=store_name)
  logging.info("File contains {} snps".format(centralized_DO.count()))

  # Split into five files
  if not os.path.isdir(split_data_dir):
    os.mkdir(split_data_dir)

#NOTE You should be able to run each block by commenting out all the other blocks except for the top block.
#NOTE Almost all of these blocks can be performed in a low memory state except for the PCA block.

########################################## QC pipeline 
  # For each filter, we run the decentralized version and compare the results to PLINK1.9 output

  backup(pheno_file, os.path.join(split_data_dir, pheno_file), [".bed", ".bim", ".fam"])
  backup('data', SCRATCH)
  os.chdir(split_data_dir)
#### Split the dataset into parts for decentralized analysis
  names = dset_split(store_name, 5, 2270, os.path.join(split_data_dir,'DO'))
  hub = analytics.Center(names)
  hub.change_pheno(pheno_file)
  backup(split_data_dir, "_backup/", ext=['.h5py', '.h5'])
  hub.loci_missing_rate_filter(0.05)
  backup(split_data_dir, "_backup_missing/", ext=['.h5py', '.h5'])
  qc_compare(names[0], plink + " --bfile ../data/popres_European --not-chr 23 25 --geno 0.05 --make-bed", "geno05")
  hub.MAF_filter(0.05)
  backup(split_data_dir, "_backup_MAF/", ext=[".h5py", ".h5"])
  qc_compare(names[0], plink + " --bfile geno05 --maf 0.05 --make-bed", "geno05maf05")
  hub.HWE_filter(1e-10)
  backup(split_data_dir, "_backup_HWE/", ext=['.h5py', '.h5'])
  qc_compare(names[0], plink + " --bfile geno05maf05 --hwe 1e-10 --make-bed", "geno05maf05hwe10")
  hub.correct_LD_prune(0.2, 50)
  runcmd(plink + " --bfile geno05maf05hwe10 --indep-pairwise 50 25 .2 --make-bed --out intermediate")
  qc_compare(names[0], plink + " --bfile geno05maf05hwe10 --exclude intermediate.prune.out --make-bed", "geno05maf05hwe10indppair502505")
  backup(split_data_dir, "_backup_pruned/", ext=['.h5py', '.h5'])
  hub.normalize()
  backup(split_data_dir, os.path.join(cwd, global_loc))


########################################## meta

  # Filter based on local_missing, local_MAF, local_LD , do local PCA
  # Transform, run local regression, average results 

  base = '_backup_MAF/'
  metaOutFile = 'meta_five_betas.txt'
  metaFiles = "meta_analysis/"
  metaFiles_regression = "meta_analysis_regression/"
  backup(base, metaFiles, ext=['.h5py', '.h5'])
  hub_names = [os.path.join(metaFiles, name) for name in os.listdir(metaFiles)]
  hub = analytics.Center(hub_names)
  hub.run_meta_filters(t_missing=0.05, t_AF=None, t_hwe=None, t_LD=None, global_clean=True)
  logging.info(compare(hub_names[0], "geno05"))
  backup(base, metaFiles, ext=['.h5py', ".h5"])
  hub.run_meta_filters(t_missing=0.05, t_AF=0.05, t_hwe=None, t_LD=None, global_clean=True)
  logging.info(compare(hub_names[0], "geno05maf05"))
  hub.run_meta_filters(t_hwe=1e-10, global_clean=True)
  logging.info(compare(hub_names[0], "geno05maf05hwe10"))
  hub.run_meta_filters(t_missing=None, t_AF=None, t_hwe=None, t_LD=.2, global_clean=True)
  logging.info(compare(hub_names[0], "geno05maf05hwe10indppair502505"))

  backup(os.path.join(split_data_dir,  "meta_analysis/"), os.path.join(cwd, global_loc))
  hub.run_local_pca(n_components=5)

  backup(base, metaFiles_regression, ext=['.h5py', '.h5'])
  hub_names = [os.path.join(metaFiles_regression, name) for name in os.listdir(metaFiles_regression)]
  hub = analytics.Center(hub_names)
  hub.copy_pca(metaFiles, local=True)
  hub.run_meta_regression(5, metaOutFile)
  backup(metaOutFile, os.path.join(cwd, metaOutFile))
  backup(split_data_dir, os.path.join(cwd, global_loc))


 #NOTE: I'll keep this next two blocks commented out. The block below requires ~45GB of RAM
 #NOTE but the regression plot can be performed on a low memory machine
########################################### PCA
#### get large machine 
#  if os.path.exists(SCRATCH): 
#    rmtree(SCRATCH)
#  loc =  os.path.join(cwd, global_loc, "five_way")
#  backup(os.path.join(cwd, loc), SCRATCH)
#  os.chdir(split_data_dir)
#  names = glob.glob('DO*.h5py')
#  hub = analytics.Center(names)
#  chroms =  None #
#  hub.PCA(chroms=chroms, n_components=5)
#  backup(split_data_dir, "_backup_PCA/", ext=['.h5py', '.h5'])
#  backup(os.path.join(split_data_dir, "_backup_PCA/"), os.path.join(cwd, loc))
#  backup(split_data_dir, global_loc, ext=['.h5py', '.h5'])
#  rmtree(SCRATCH)
#
########################################### regression 
### PUll in the entire data since we deleted stuff 

#  Dbeta = "five_D_betas.txt"
#  Dbeta_avg = "five_D_betas_avg.txt"
#  if os.path.exists(SCRATCH):
#    rmtree(SCRATCH)
#  backup(os.path.join(global_loc, "five_way"), SCRATCH)
#  os.chdir(split_data_dir)
#
#  base = '_backup_MAF/'
#  regression_dir = '_regression/'
#  backup(base, regression_dir, ext=['.h5py', '.h5'])
#  data_names = [os.path.join(regression_dir, fname) for fname in os.listdir(regression_dir)]
#
#  hub = analytics.Center(data_names)
#  hub.copy_pca(os.path.join(split_data_dir, '_backup_PCA'))
#  hub.normalize()
#  backup(os.path.join(split_data_dir, "_regression/"), os.path.join(cwd, global_loc, "five_way"))
#
#  hub.run_regression(5, 100, out_file=Dbeta)
#  logging.info("GWAS is finished!")
#
#  logging.info("GWAS is via AVG!")
#  hub.run_regression(5, 1, kind='AVG', out_file=Dbeta_avg)
#  logging.info("GWAS via AVG finished!")
#  
#  backup(Dbeta, os.path.join(cwd, Dbeta))
#  backup(Dbeta, os.path.join(cwd, Dbeta_avg))
#
###### Timing
#  iterName = "five_D_betas_iters.txt"
#  hub.run_regression(5, 100, chroms=[17], verbose=True, out_file=iterName)
#  logging.info("Finished timing experiment")
#  backup(iterName, os.path.join(cwd, iterName))



### Check timing
#  full_hub.run_regression(5, 100, chroms=[22], verbose=True, out_file="Decentralized_22_betas_iters.txt")

if __name__=='__main__':
  run_experiment()

