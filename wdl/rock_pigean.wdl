version 1.0

workflow rock_pigean {
    input {
        File gene_list
        String output_files_base_name
    }
    call run_pigean {
        input: gene_list=gene_list, output_files_base_name=output_files_base_name
    }
    output {
        File gs = run_pigean.gs
        File gss = run_pigean.gss
        File ggss = run_pigean.ggss
        File params = run_pigean.params
    }
}

task run_pigean {
    input {
        File gene_list
        String output_files_base_name
    }
    runtime {
        docker: "gcr.io/nitrogenase-docker/rock-pigean:1.0.0"
    }
    command <<<
        python3 -u /app/priors.py gibbs \
        --X-in /app/data/gene_set_list_mouse_2024.txt \
        --X-in /app/data/gene_set_list_msigdb_nohp.txt \
        --gene-map-in /app/data/portal_gencode.gene.map \
        --max-num-gene-sets 5000 \
        --gene-stats-out gs.~{output_files_base_name}.out \
        --gene-set-stats-out gss.~{output_files_base_name}.out \
        --gene-gene-set-stats ggss.~{output_files_base_name}.out \
        --params-out p.~{output_files_base_name}.out \
        --debug-level 3 \
        --positive-controls-in ~{gene_list} \
        --gene-loc-file /app/data/NCBI37.3.plink.gene.loc \
        --gene-loc-file-huge /app/data/refGene_hg19_TSS.subset.loc \
        --exons-loc-file-huge /app/data/NCBI37.3.plink.gene.exons.loc \
        --gene-filter-value 1 --gene-set-filter-value 0.01
    >>>
    output {
        File gs = "gs." + output_files_base_name + ".out"
        File gss = "gss." + output_files_base_name + ".out"
        File ggss = "ggss." + output_files_base_name + ".out"
        File params = "p." + output_files_base_name + ".out"
    }
}