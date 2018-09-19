# Armin Pourshafeie

# Reads Chiamo files and parses them into a numpy datastore
import os
import logging
import numpy as np
import gzip, h5py
import pdb
import re 
import sys
from plinkio import plinkfile
import gc


def reader(fp):
  line = fp.readline()
  if len(line) > 0:
    dummy = line[:-2].split(' ')[5:]
    num_inds = len(dummy) + 5

  while len(line) > 0:   # Poor mans' end of file detection
    line = line[:-2].split(' ')
    snp, rsid, pos = line[0], line[1], line[2]
    gts = np.array([float(line[i + 1]) + float(line[i + 2]) * 2 for i in range(5, num_inds,3)])
    yield snp, rsid, pos, gts
    line =  fp.readline()


def files_to_read(loc_fname):
  logging.info("-Reading data files from: " + loc_fname)
  all_to_reads = []
  directories = []
  with open(loc_fname, 'r') as fp:
    lines = fp.readlines()
  lines = [line[:-1] for line in lines] # strip newline char
  n_tot = int(lines[0])
  logging.info("-Will analyze {} individuals".format(n_tot))
  # loop over the file and read/write the data
  for ln in lines[1:]:
    ln = ln.split(',')
    directory = ln[0]
    directories.append(directory)
    dirList = os.listdir(directory)
    toRead = sorted([(os.path.join(directory,item), ln[1])
      for item in dirList if item[-9:] == 'chiamo.gz'])
    all_to_reads.append(toRead)
    all_to_reads = [item for group in all_to_reads for item in group]
  return n_tot, all_to_reads, directories

def get_chrom(filename):
  """Given a file path tries to guess the chromsome. 
  This just looks for digits so make sure the path name doesn't have
  other digits in it"""
  basename = os.path.basename(filename)
  pattern = re.compile("[0-2][0-9]")
  m = pattern.search(basename)
  try:
    chrm = m.group(0)
  except AttributeError:
    logging.error("Couldn't find the Chromosome name :(")
    sys.exit("Failed at finding the chromosome name")
  return chrm


def incorporate_demography(directory, store):
  
  # Try guessing the folders name. (file name must contain 
  # sample, 200 and txt in it 
  demographic_file = ''
  for fname in os.listdir(directory):
    if fname == 'Affx_sample_NBS.txt':
      demographic_file = fname
  demographic_f = os.path.join(directory, demographic_file)
  region = []
  wtid   = []
  with open(demographic_f, 'r') as dem_f:
    for line in dem_f:
      if line[0] == '#': #skip header
        continue 
      line = line[:-1]  # get rid of \n 
      splitted = re.split('\t| ', line)
      region.append(splitted[5])
      wtid.append(splitted[0])
  return np.array(region), np.array(wtid)



def shuffled_list(data_order, seed):
  np.random.seed(seed)
  np.random.shuffle(data_order)
  return data_order

def read_datasets(loc_fname, shuffle, store_name='test.h5', seed=25):
  n_tot, to_read, directories= files_to_read(loc_fname)
  data_order = range(n_tot)
  if shuffle:
    data_order = shuffled_list(data_order, seed)

  with h5py.File(store_name, 'w-', libver='latest', swmr=True) as store:
    store.attrs['has_local_AF'] = False
    store.attrs['normalized']   = False
    filled, dir_num = 0, 0
    for filepath, status in to_read:
      status = status
      directory = directories[dir_num]
      logging.debug("--Working on: " + filepath)
      chrom = get_chrom(filepath)
      ordered = False
      # see if there are new individuals
      if directory not in filepath:
        # Logically this should be executed once per every cohort
        filled += len(gts)
        dir_num += 1
        # we are done with this directory! write the status!
        dset = store.require_dataset('meta/Status', (n_tot,), dtype=np.int8)
        dset[to_fill,] = status
        regions, wtid = np.array(incorporate_demography(directory, store))
        dset = store.require_dataset('meta/regions', (n_tot,), dtype ='S19')
        dset[to_fill,] = regions[order]
        dset = store.require_dataset('meta/id', (n_tot,), dtype='S11')
        dset[to_fill,] = wtid[order]
      with gzip.open(filepath, 'rb') as file_pointer:
        current_group = store.require_group(chrom)
        gen = reader(file_pointer)
        i = 0
        for snp, rsid, pos, gts in gen:
#          if i % 5 != 0:
#            i += 1
#            continue
          if not ordered:
            to_fill = np.array(data_order[filled:filled+len(gts)])
            order = np.argsort(to_fill)
            to_fill = range(len(to_fill))#to_fill[order] #TODO suspect
            ordered = True
          #to_fill, gts = zip(*sorted(zip(to_fill, gts))) #TODO fix this so 
          gts = gts[order]
           # We don't have to sort every time
          dset = current_group.require_dataset(pos, (n_tot,), dtype=np.float32)
          # check to make sure ref/alts/rsids are not screwed up
          if 'rsid' in dset.attrs:
            if dset.attrs['rsid'] != rsid:
              sys.exit("rsid's don't match for chrom {}. pos {}".format(
                chrom, pos))
          else:
            dset.attrs['rsid'] = rsid
            dset.attrs['snp'] = snp
          dset[to_fill,] = gts
#          if i % 5000==0:
#            print i
#          i += 1
#          if i > 10:
#            break
        # end of generator loop
      # end of context manager for filepath
    # end of to_read loop
    dset = store.require_dataset('meta/Status', (n_tot,), dtype=np.int8)
    dset[to_fill,] = float(status)
    regions, wtid = incorporate_demography(directory, store)
    dset = store.require_dataset('meta/regions', (n_tot,), dtype ='S19')
    dset[to_fill,] = regions[order]
    dset = store.require_dataset('meta/id', (n_tot,), dtype='S11')
    dset[to_fill,] = wtid[order]

#@profile
def read_plink(loc_fname, dem_file, shuffle, store_name='test.h5', seed=25, n_tot=None, start=0):
  logging.info("-Reading plink file at: {}".format(loc_fname))
  plink_file = plinkfile.open(loc_fname)
  if not plink_file.one_locus_per_row():
    print( "This script requires that snps are rows and samples columns." )
    sys.exit(1)

  sample_list = plink_file.get_samples()
  locus_list = plink_file.get_loci()
  if n_tot is None:
    n_tot = len(sample_list)
  data_order = range(n_tot)
  if shuffle:
    data_order = shuffled_list(data_order, seed)
  
  with h5py.File(store_name, 'w', libver='latest', swmr=True) as store:
    store.attrs['has_local_AF'] = False
    store.attrs['normalized']   = False
    to_fill = data_order[start:start + len(sample_list)]
    order = np.argsort(to_fill)
    to_fill = list(np.sort(to_fill))
    assert to_fill == range(len(sample_list)) 
    dset = store.require_dataset('meta/Status', (n_tot,), dtype=np.int8)
    dset[to_fill,] = [sample_list[i].affection for i in order]
    with open(dem_file, 'r') as dem_f:
      dem_f.next()
      pop_dict = {line.split('\t')[0]: line.split('\t')[5] for line in dem_f}
      #pop = np.array([line.split('\t')[5].rstrip() for line in dem_f])

    dset = store.require_dataset('meta/id', (n_tot,), dtype='S11')
    dset[to_fill,] = [sample_list[i].iid for i in order]
    
    ids = [sample_list[i].iid for i in order]
    countries = [pop_dict[iid] for iid in ids]
    del ids
    dset = store.require_dataset('meta/regions', (n_tot,), dtype='S19')
    dset[to_fill,] = countries
    current_chrom = 1
    current_group = store.require_group(str(current_chrom))
#    i = 0
    #for locus, row in zip(locus_list, plink_file):
    genotypes = np.zeros(len(order), dtype=np.int8)
    for locus in locus_list:
      row = next(plink_file)
      if locus.chromosome != current_chrom:
        logging.info("--Finished adding chromosome: {}".format(current_chrom))
        current_chrom = locus.chromosome
        if current_chrom == 23:
          break
        current_group = store.require_group(str(current_chrom))
      genotypes[:] = np.array(row, dtype=np.int8)[order]
      pos = str(locus.bp_position)
      dset = current_group.require_dataset(pos, (n_tot,), dtype=np.float32)
      dset[:,] = genotypes
      dset.attrs['rsid'] = locus.name
      dset.attrs['snp']  = locus.allele1
      dset.attrs['alt'] = locus.allele2
#      if i%50000 == 0:
#        #gc.collect()
#        sys.exit()
#      i += 1



#  for locus, row in zip(locus_list, plink_file):
#    for sample, genotype in zip(sample_list, row):






if __name__=='__main__':
  plink_file = '../EV/POPRES/plink/EURthinnedPOPRES+Spain6.0_clean'
  read_plink(plink_file, plink_file[:-5]+'demodata.txt', True)
#  with gzip.open('../WTCCC_data/EGAD00000000002/Oxstat_format/NBS_04_chiamo.gz', 'rb') as fp:
#    gen = reader(fp)
#
#    i = 0
#    for snp, rsid, pos, item in gen:
#      pdb.set_trace()
##      print item 
#      i += 1
#      print i
#      if i > 3:
#        break
#


