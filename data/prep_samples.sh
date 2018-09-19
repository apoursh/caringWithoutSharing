plink="/srv/gsfs0/software/plink/1.90/plink"
data="/srv/gsfs0/projects/bustamante/reference_panels/PopRes/ALL/GSK_090212_forwardstrandFull"
home="/srv/gsfs0/projects/bustamante/apoursh_projects/Decentralized/cleanedData/data"
headName="popres_European"
excludeInds="toExclude.txt"

# Get European individuals
head -1 popres_demodata.txt > $headName.ind
cat popres_demodata.txt | grep "European" | grep -ve "Mix\|EuropeanA\|EuropeanB" >> ${headName}.ind

# limit to Europeans
$plink --bfile $data  --keep popres_European.ind  --make-bed  --out $home/tmp 

# exclude outliers 
$plink --bfile tmp  --remove-fam $excludeInds --not-chr 23 25 --mind 0.05 --make-bed  --out $home/$headName
rm tmp.bed tmp.bim tmp.fam


$plink --bfile $home/$headName  --geno 0.05  --maf 0.05 --make-bed  --out $home/${headName}geno05maf05

#PCA ready
$plink --bfile $home/${headName}geno05maf05  --hwe 1e-10  --make-bed  --out $home/${headName}hwe
$plink --bfile $home/${headName}hwe --indep-pairwise 50 25 .2  --make-bed  --out intermediate

$plink --bfile $home/${headName}hwe --exclude intermediate.prune.out  --make-bed  --out $home/${headName}geno05maf05hwe10indppair502502
