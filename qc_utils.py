# A bunch of functions that perform qc using plink and or compare 
import numpy as np 
from plinkio import plinkfile
import subprocess
import logging
import h5py
import pdb
import os
import shutil


def qc_compare(filename, cmd, cmd_out):
  cmd = cmd + """ --out {}""".format(cmd_out)
  print(cmd)
  process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()
  logging.info(output)
  logging.info(error)
  intersection, length_of_p, length_of_hfile = compare(filename, cmd_out)
  logging.info("""File produced by command: {cmd}
  has {intersection} intersecting loci with the file {hfile}
  This is {percent}% overlap. And the python set has size {hset}""".format(cmd=cmd, intersection=intersection,
    hfile=filename, percent=100*float(intersection)/max(length_of_p, length_of_hfile), hset=length_of_hfile))

  

def compare(hdf_file, plink_file):
  pfile = plinkfile.open(plink_file)
  if not pfile.one_locus_per_row():
    logging.error("""This script requires the snps to be rows and samples to be columns.""")
    sys.exit(1)

  locus_list = pfile.get_loci( )
  pset = { (l.chromosome, l.bp_position) for l in locus_list }
#  pset = {item for item in pset if item[0] == 1}#TODO remove
  total_intersection = 0
  total_hset_length = 0
  total_pset_length = len(pset)
  with h5py.File(hdf_file, "r") as hfile: 
    for key in hfile.keys():
      if key == "meta":
        continue 
      ikey = int(key)
      hset = {(ikey, int(pos)) for pos in hfile[key].keys()}
#      testpset= {(i[0], i[1]) for i in pset if i[0] == ikey}
#      if len(testpset - hset) > 0 or len( hset - testpset) > 0:
#        pdb.set_trace()
      pset_len = len(pset)
#      hmp = sorted([i for _,i in hset - pset])
#      pmh = sorted([i for _,i in pset - hset])
      pset = pset - hset
      total_intersection += pset_len - len(pset)
      total_hset_length += len(hset)

  return(total_intersection, total_pset_length, total_hset_length)


def runcmd(cmd):
  process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()
  logging.info(output)

def copy_files(src, dest):
  if not os.path.isdir(dest):
    os.mkdir(dest)
  if os.path.isdir(src):
    src_files = os.listdir(src)
    for file_name in src_files:
      full_src_name = os.path.join(src, file_name)
      if (os.path.isfile(full_src_name)):
        full_dest_name = os.path.join(dest, file_name)
        shutil.copy(full_src_name, full_dest_name)
  else:
    shutil.copy(src, dest)

