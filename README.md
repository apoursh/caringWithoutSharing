# Caring Without Sharing


A complete pipeline for performing GWAS in a decentralized fashion. We perform QC, PCA and logistic regression in two scenarios. In scenario 1, we randomly split the data into five silos. In scenario 2, we split the data based on case control status. In both scenarios, we compare the results to a centralized analysis and demonstrate that almost perfect numerical agreement can be achieved. We furthermore, investigate our performance on extremely rare alleles and show that out pipeline can achieve the accuracy of a centralized pipeline when the allele count per silo is too low for meta studies to perform well.
You can find our preprint <a href="noLinkYet">here</a>. This work is currently under submission. 


## Included files: 
<ul>
  <li>Plots_Analysis.ipynb: Notebook with the analysis results for scenario 1, 2.</li>
  <li>rare_allele_sim.ipynb: Code/plots analysis of FE vs our method for very rare alleles.</li>
  <li>add_pheno.py: Simulates a phenotype and performs a local GWAS.</li>
  <li>scenario1.py: Code for generating the data for scenario 1.</li>
  <li>scenario2.py: Code for generating the data for scenario 2.</li>
  <li>data/prep_samples.sh: PLINK pipeline to subset the data for simulated GWAS.</li>
  <li>data/toExclude.txt: Outlier individuals excluded from the analysis.</li>
  <li>optimizationAux.py: Auxiliary files for running lbfgsb for admm regression.</li>
  <li>corr.pyx: fast correlation compuatation. Same method as used by PLINK. </li>
  <li>qc_utils.py: Utilities for running PLINK commands and comparing hdf stores with plink outputs</li>
  <li>process_chiamo.py: Utility functions for reading PLINK/chiamo files to hdf (chiamo files were not used in our experiments)</li>
</ul>
