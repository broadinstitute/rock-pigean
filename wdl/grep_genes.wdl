version 1.0

workflow grep_genes {
    input {
        File gene_list
        File data
        String output_file_name
    }
    call run_grep_genes {
        input: gene_list=gene_list, data=data, output_file_name=output_file_name
    }
}

task run_grep_genes {
    input {
        File gene_list
        File data
        String output_file_name
    }
    runtime {
        docker: "ubuntu:latest"
    }
    command <<<
        for gene in $(cat ~{gene_list}); do
            { echo -n \"gene\": \"; echo -n "${gene}" | tr -d [:cntrl:]; echo \"; } >> patterns;
        done
        grep -f patterns ~{data} > ~{output_file_name}
    >>>
    output {
        File output_file = output_file_name
    }
}