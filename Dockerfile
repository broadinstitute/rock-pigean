FROM ubuntu:plucky
RUN apt -y update && apt -y upgrade && apt -y install python3 python3-scipy
ADD data/gene_set_list_mouse_2024.txt data/
ADD data/gene_set_list_msigdb_nohp.txt data/
ADD data/portal_gencode.gene.map  data/
ADD data/NCBI37.3.plink.gene.loc  data/
ADD data/refGene_hg19_TSS.subset.loc  data/
ADD data/NCBI37.3.plink.gene.exons.loc data
ADD priors.py priors.py
