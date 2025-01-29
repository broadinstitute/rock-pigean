python -u /home/unix/flannick/lap/projects/ldsc/bin/priors.py gibbs \
 --X-in /humgen/diabetes2/users/lthakur/lap_test/ldsc/results/out/projects/euro/gene_set_lists/gene_set_list_mouse_2024.txt \
 --X-in /humgen/diabetes2/users/lthakur/lap_test/ldsc/results/out/projects/euro/gene_set_lists/gene_set_list_msigdb_nohp.txt \
 --gene-map-in /humgen/diabetes2/users/lthakur/lap_test/ldsc/raw/portal_gencode.gene.map \
 --max-num-gene-sets 5000 \
 --gene-stats-out gs.out --gene-set-stats-out gss.out --gene-gene-set-stats ggss.out --params-out p.out --debug-level 3 \
 --positive-controls-in /path/to/file/no/header/one_column/gene_list \
 --gene-loc-file /humgen/diabetes2/users/lthakur/lap_test/ldsc/raw/NCBI37.3.plink.gene.loc \
 --gene-loc-file-huge /humgen/diabetes2/users/lthakur/lap_test/ldsc/raw/refGene_hg19_TSS.subset.loc \
 --exons-loc-file-huge /humgen/diabetes2/users/lthakur/lap_test/ldsc/raw/NCBI37.3.plink.gene.exons.loc \
 \
 --gene-filter-value 1 --gene-set-filter-value 0.01 \
 >& log