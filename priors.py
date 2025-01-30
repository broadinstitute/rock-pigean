#Usage: python calc.py
#
#This script...
#
#Arguments:
# warnings-file[path]: file to output warnings to; default: standard error
#these are for debugging
#import cProfile
#import resource

#import urllib.request #used (below) only if you specify a URL as an input file
#import requests #used (below) only if you specify --lmm-auth-key

import optparse
import sys
import time
import os
import copy
import scipy
import scipy.sparse as sparse
import scipy.stats
import numpy as np
import itertools
import gzip
import random

random.seed(0)

def bail(message):
    raise ValueError(message)
    sys.stderr.write("%s\n" % (message))
    sys.exit(1)

def get_current_memory_usage_linux(tag=None):
    with open('/proc/self/status') as f:
        for line in f:
            if 'VmRSS' in line:  # Resident Set Size
                memory_kb = int(line.split()[1])  # Memory in KB
                if tag is not None:
                    print(tag)
                print(f"Current memory usage: {memory_kb / 1024:.2f} MB")
                get_memory_usage()
                return memory_kb / 1024

def get_memory_usage():
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Max RSS in KB
    print(f"Max memory usage: {usage / 1024:.2f} MB")


usage = "usage: priors.py [beta_tildes|sigma|betas|priors|naive_priors|gibbs|factor|naive_factor|sim|pops|naive_pops] [options]"

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
def get_comma_separated_args_as_set(option, opt, value, parser):
    setattr(parser.values, option.dest, set(value.split(',')))

parser = optparse.OptionParser(usage)
#gene x gene_set matrix
#each specification of these files is a different batch
#can use "," to group multiple files or lists within each --X
#can combine into batches with "@{batch_id}" after the file/list
#by default, the same @{batch_id} is appended to a list, which meansit will be appended to all files in the list that do not already have a batch
#this can be overriden by specifying batches to files within the list
#these batches are used for parameter learning (see below)
parser.add_option("","--X-in",action="append",default=None)
parser.add_option("","--X-list",action="append",default=None)
parser.add_option("","--Xd-in",action="append",default=None)
parser.add_option("","--Xd-list",action="append",default=None)
parser.add_option("","--X-out",default=None)
parser.add_option("","--Xd-out",default=None)
parser.add_option("","--ignore-genes",action='append',default=["NA"]) #gene names to ignore
parser.add_option("","--batch-separator",default="@") #separator for batches
parser.add_option("","--file-separator",default=None) #separator for multiple files

#model parameters
parser.add_option("","--p-noninf",type=float,default=None) #initial parameter for p
parser.add_option("","--sigma2-cond",type=float,default=None) #specify conditional sigma value (sigma/p). Precedence 1
parser.add_option("","--sigma2-ext",type=float,default=None) #specify sigma in external units. Precedence 2
parser.add_option("","--sigma2",type=float,default=None) #specify sigma in internal units (this is what the code outputs to --sigma-out). Precedence 3
parser.add_option("","--top-gene-set-prior",type=float,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the top prior across all gene sets. Precedence 4
parser.add_option("","--num-gene-sets-for-prior",type=int,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the either the number of non-zero gene sets (by default) or the total number of gene sets (if --frac-gene-sets-for-prior is set to a number below 1).  Precedence 4
parser.add_option("","--frac-gene-sets-for-prior",type=float,default=1) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). If this is changed from its default value of 1, it will fit sigma-cond from top and num, and then convert to (internally stored) total var. Precedence 4
parser.add_option("","--sigma-power",type='float',default=None) #multiply sigma times np.power(scale_factors,sigma_power). 2=const_sigma, 0=default. Larger values weight larger gene sets more
parser.add_option("","--sigma-soft-threshold-95",type='float',default=None) #the gene set size at which threshold is 0.95
parser.add_option("","--sigma-soft-threshold-5",type='float',default=None) #the gene set size at which threshold is 0.05


parser.add_option("","--const-sigma",action='store_true') #assign constant variance across all gene sets independent of size (default is to scale inversely to size). Overrides sigma power and sets it to 2

parser.add_option("","--update-hyper",type='string',default=None,dest="update_hyper") #update either both,p,sigma,none
parser.add_option("","--cross-val",action='store_true',dest="cross_val",default=None) #after initial learning of p and sigma, do cross validation to tune sigma further
parser.add_option("","--no-cross-val",action='store_false',dest="cross_val",default=None) #after initial learning of p and sigma, do cross validation to tune sigma further
parser.add_option("","--cross-val-num-explore-each-direction",type='int',default=3) #the number of orders of magnitude canges to try cross validation for
parser.add_option("","--cross-val-max-num-tries",type='int',default=2) #if the best cross validation result is a boundary, then re-explore further in that direction. Repeat this many times
parser.add_option("","--cross-val-folds",type='int',default=4) #the number of orders of magnitude canges to try cross validation for
parser.add_option("","--sigma-num-devs-to-top",default=2.0,type=float) #update sigma based on top gene set being this many devs away from zero
parser.add_option("","--p-noninf-inflate",default=1.0,type=float) #update p by multiplying it by this each time you learn it

parser.add_option("","--batch-all-for-hyper",action="store_true") #combine everything into one batch for learning hyper
parser.add_option("","--first-for-hyper",action="store_true") #use first batch / dataset (that is, the batch of the first --X; may include other files too) to learn parameters for unlabelled batches (label batches with "@{batch_id}" as abov)
parser.add_option("","--first-for-sigma-cond",action="store_true") #use first batch to fix sigma/p ratio and use that for all other batches. 
parser.add_option("","--first-max-p-for-hyper",action="store_true") #use first batch / dataset (that is, the batch of the first --X; may include other files too) to learn the maximum parameters for unlabelled batches (label batches with "@{batch_id}" as above)

parser.add_option("","--background-prior",type=float,default=0.05) #specify background prior

#correlation matrix (otherwise will be calculated from X)
parser.add_option("","--V-in",default=None)
parser.add_option("","--V-out",default=None)
parser.add_option("","--shrink-mat-out",default=None)

#optional gene name map
parser.add_option("","--gene-map-in",default=None)
parser.add_option("","--gene-map-orig-gene-col",default=1) #1-based column for original gene
parser.add_option("","--gene-map-new-gene-col",default=2) #1-based column for original gene

#GWAS association statistics (for HuGECalc)
parser.add_option("","--gwas-in",default=None)
parser.add_option("","--gwas-locus-col",default=None)
parser.add_option("","--gwas-chrom-col",default=None)
parser.add_option("","--gwas-pos-col",default=None)
parser.add_option("","--gwas-p-col",default=None)
parser.add_option("","--gwas-beta-col",default=None)
parser.add_option("","--gwas-se-col",default=None)
parser.add_option("","--gwas-units",type=float,default=None)
parser.add_option("","--gwas-n-col",default=None)
parser.add_option("","--gwas-n",type='float',default=None)
parser.add_option("","--gwas-freq-col",default=None)
parser.add_option("","--gwas-filter-col",default=None) #if specified, only include rows of the gwas file where this column matches --gwas-filter-val
parser.add_option("","--gwas-filter-value",default=None) #if specified, only include rows of the gwas file where this value is observed in --gwas-filter-col
parser.add_option("","--gwas-ignore-p-threshold",type=float,default=None) #completely ignore anything with p above this threshold

#credible sets
parser.add_option("","--credible-sets-in",default=None) #pass in credible sets to use 
parser.add_option("","--credible-sets-id-col",default=None)
parser.add_option("","--credible-sets-chrom-col",default=None)
parser.add_option("","--credible-sets-pos-col",default=None)
parser.add_option("","--credible-sets-ppa-col",default=None)

#S2G values (for HuGeCalc)
parser.add_option("","--s2g-in",default=None)
parser.add_option("","--s2g-chrom-col",default=None)
parser.add_option("","--s2g-pos-col",default=None)
parser.add_option("","--s2g-gene-col",default=None)
parser.add_option("","--s2g-prob-col",default=None)
parser.add_option("","--s2g-normalize-values",type=float,default=None) #for each variant, set sum of probabilities across genes to be equal to this value. Relative values are kept the same

#Exomes association statistics (for HuGeCalc)
parser.add_option("","--exomes-in",default=None)
parser.add_option("","--exomes-gene-col",default=None)
parser.add_option("","--exomes-p-col",default=None)
parser.add_option("","--exomes-beta-col",default=None)
parser.add_option("","--exomes-se-col",default=None)
parser.add_option("","--exomes-units",type=float,default=None)
parser.add_option("","--exomes-n-col",default=None)
parser.add_option("","--exomes-n",type='float',default=None)

#Positive control genes
parser.add_option("","--positive-controls-in",default=None)
parser.add_option("","--positive-controls-id-col",default=None)
parser.add_option("","--positive-controls-prob-col",default=None)
parser.add_option("","--positive-controls-list",type="string",action="callback",callback=get_comma_separated_args,default=None) #specify comma separated list of positive controls on the command line
parser.add_option("","--positive-controls-default-prob",type=float,default=0.95)
parser.add_option("","--positive-controls-no-header",action="store_false", dest="positive_controls_has_header", default=True)
parser.add_option("","--positive-controls-all-in",default=None) #all genes to use in positive control analysis. If specified add these on top of the positive controls
parser.add_option("","--positive-controls-all-id-col",default=None)
parser.add_option("","--positive-controls-all-no-header",action="store_false", dest="positive_controls_all_has_header", default=True)


#association statistics for gene bfs in each gene set (if precomputed)
#REMINDER: the betas are all in *external* units
parser.add_option("","--gene-set-stats-in",default=None)
parser.add_option("","--gene-set-stats-id-col",default="Gene_Set")
parser.add_option("","--gene-set-stats-exp-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-col",default=None)
parser.add_option("","--gene-set-stats-beta-uncorrected-col",default=None)
parser.add_option("","--gene-set-stats-se-col",default=None)
parser.add_option("","--gene-set-stats-p-col",default=None)
parser.add_option("","--ignore-negative-exp-beta",action='store_true')

#if you have gene set betas
parser.add_option("","--gene-set-betas-in",default=None)
parser.add_option("","--const-gene-set-beta",default=None,type=float)

#gene BFs to use in calculating gene set statistics
parser.add_option("","--gene-bfs-in",default=None)
parser.add_option("","--gene-stats-in",dest="gene_bfs_in",default=None)
parser.add_option("","--gene-bfs-id-col",default=None)
parser.add_option("","--gene-stats-id-col",default=None,dest="gene_bfs_id_col")
parser.add_option("","--gene-bfs-log-bf-col",default=None)
parser.add_option("","--gene-stats-log-bf-col",default=None,dest="gene_bfs_log_bf_col")
parser.add_option("","--gene-bfs-combined-col",default=None)
parser.add_option("","--gene-stats-combined-col",default=None,dest="gene_bfs_combined_col")
parser.add_option("","--gene-bfs-prior-col",default=None)
parser.add_option("","--gene-stats-prior-col",default=None,dest="gene_bfs_prior_col")
parser.add_option("","--gene-bfs-prob-col",default=None)
parser.add_option("","--gene-stats-prob-col",default=None,dest="gene_bfs_prob_col")
parser.add_option("","--const-gene-log-bf",default=None,type=float)

#gene percentiles to use in calculating gene set statistics. Will be converted to BFs using inverse normal function
parser.add_option("","--gene-percentiles-in",default=None)
parser.add_option("","--gene-percentiles-id-col",default=None)
parser.add_option("","--gene-percentiles-value-col",default=None)
parser.add_option("","--gene-zs-in",default=None)
parser.add_option("","--gene-zs-id-col",default=None)
parser.add_option("","--gene-zs-value-col",default=None)

#locations of genes
#ALL GENE LOC FILES MUST BE IN FORMAT "GENE CHROM START END STRAND GENE" 
parser.add_option("","--gene-loc-file",default=None)
parser.add_option("","--gene-loc-file-huge",default=None)
parser.add_option("","--exons-loc-file-huge",default=None)
parser.add_option("","--gene-cor-file",default=None)
parser.add_option("","--gene-cor-file-gene-col",type=int,default=1)
parser.add_option("","--gene-cor-file-cor-start-col",type=int,default=10)

#additional covariates to use in the model
parser.add_option("","--no-correct-huge",default=True,action='store_false',dest="correct_huge") #don't correct huge scores for confounding variables. If --correct-huge, these covariates will be added on top of any extra covariates
parser.add_option("","--gene-covs-in",default=None) #extra covariates to correct Y 

#output files for stats
parser.add_option("","--gene-set-stats-out",default=None)
parser.add_option("","--gene-set-stats-trace-out",default=None)
parser.add_option("","--betas-trace-out",default=None)
parser.add_option("","--gene-stats-out",default=None)
parser.add_option("","--gene-stats-trace-out",default=None)
parser.add_option("","--gene-gene-set-stats-out",default=None)
parser.add_option("","--gene-set-overlap-stats-out",default=None)
parser.add_option("","--gene-covs-out",default=None)
parser.add_option("","--gene-effectors-out",default=None)
parser.add_option("","--phewas-stats-out",default=None)
parser.add_option("","--factors-out",default=None)
parser.add_option("","--factors-anchor-out",default=None)
parser.add_option("","--gene-set-clusters-out",default=None)
parser.add_option("","--gene-clusters-out",default=None)
parser.add_option("","--pheno-clusters-out",default=None)
parser.add_option("","--gene-set-anchor-clusters-out",default=None)
parser.add_option("","--gene-anchor-clusters-out",default=None)
parser.add_option("","--pheno-anchor-clusters-out",default=None)
parser.add_option("","--factor-phewas-stats-out",default=None)

#for pheno factoring
parser.add_option("","--gene-pheno-stats-out",default=None)

#run a phewas against the gene scores
parser.add_option("","--run-phewas-from-gene-phewas-stats-in",default=None) #specify the gene phewas stats to run a phewas against
parser.add_option("","--factor-phewas-from-gene-phewas-stats-in",default=None) #specify the gene phewas stats to run a factor phewas against

parser.add_option("","--factor-phewas-min-gene-factor-weight",type=float,default=0.01) #if genes have max weight across factors less than this, remove them before running phewas

#limit gene sets printed
parser.add_option("","--max-no-write-gene-set-beta",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-gene-set-beta",type=float,default=0) #do not write gene sets to gene-gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--use-beta-uncorrected-for-gene-gene-set-write-filter",action="store_true",default=False) #filter on beta uncorrected rather than beta when filtering gene/gene set pairs to write
parser.add_option("","--max-no-write-gene-set-beta-uncorrected",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-pheno",type=float,default=0) #write only gene-pheno pairs if one value in the row is higher than this

#output for parameters
parser.add_option("","--params-out",default=None)

#control output / logging
parser.add_option("","--log-file",default=None)
parser.add_option("","--warnings-file",default=None)
parser.add_option("","--debug-level",type='int',default=None)
parser.add_option("","--hide-progress",default=False,action='store_true')
parser.add_option("","--hide-opts",default=False,action='store_true')

#behavior of regression
parser.add_option("","--ols",action='store_true') #run ordinary least squares rather than corrected ordinary least squares
parser.add_option("","--linear",action='store_true',dest="linear",default=None) #run linear regression on odds rather than logistic regression on binary disease status. Applies only to beta_tildes and priors, not gibbs
parser.add_option("","--no-linear",action='store_false',dest="linear",default=None) #run linear regression on odds rather than logistic regression on binary disease status. Applies only to beta_tildes and priors, not gibbs
parser.add_option("","--max-for-linear",type='float',default=None) #if linear regression is specified, it will switch to logistic regression if a probability exceeds this value
parser.add_option("","--use-sampling-for-betas",type='int',default=None) #rather than taking top X% of gene sets to be positive during gene set statistics, sample from probability distribution


#other control
parser.add_option("","--hold-out-chrom",type="string",default=None) #don't use this chromosome for input values (infer only priors, based on other chromosomes)
parser.add_option("","--permute-gene-sets",action='store_true',default=None) #randomly shuffle the genes across gene sets (useful for negative controls)

#parameters for controlling efficiency
#split genes into batches for calculating final statistics via cross-validation
parser.add_option("","--priors-num-gene-batches",type="int",default=20)
parser.add_option("","--gibbs-num-batches-parallel",type="int",default=10)
parser.add_option("","--gibbs-max-mb-X-h",type="int",default=100)
parser.add_option("","--batch-size",type=int,default=5000) #maximum number of dense X columns to hold in memory at once
parser.add_option("","--pre-filter-batch-size",type=int,default=None) #if more than this number of gene sets are about to go into non inf betas, do pre-filters on smaller batches. Assumes smaller batches will only have higher betas than full batches
parser.add_option("","--pre-filter-small-batch-size",type=int,default=500) #the limit to use for the smaller pre-filtering batches
parser.add_option("","--max-allowed-batch-correlation",type=float,default=0.5) #technically we need to update each gene set sequentially during sampling; for efficiency, group those for simultaneous updates that have max_allowed_batch_correlation below this threshold
parser.add_option("","--no-initial-linear-filter",default=True,action="store_false",dest="initial_linear_filter") #within gibbs sampling, first run a linear regression to remove non-associated gene sets (reducing number that require full logistic regression)

#parameters for filtering gene sets
parser.add_option("","--min-gene-set-size",type=int,default=None) #ignore genes with fewer genes than this (after removing for other reasons)
parser.add_option("","--filter-gene-set-p",type=float,default=None) #gene sets with p above this are never seen. If this is above --max-gene-set-p, then it will be lowered to match --max-gene-set-p
parser.add_option("","--filter-negative",default=None,action="store_true",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)
parser.add_option("","--no-filter-negative",default=None,action="store_false",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)

parser.add_option("","--increase-filter-gene-set-p",type=float,default=0.01) #require at least this fraction of gene sets to be kept from each file
parser.add_option("","--max-num-gene-sets-initial",type=int,default=None) #ignore gene sets to reduce to this number. Uses nominal p-values. Happens before expensive operations (pruning, parameter estimation, non-inf betas)
parser.add_option("","--max-num-gene-sets",type=int,default=None) #ignore gene sets to reduce to this number. Uses pruning to find independent gene sets with highest betas. Happens afer expensive operations (pruning, parameter estimation) but before gibbs
parser.add_option("","--min-num-gene-sets",type=int,default=1) #increase filter_gene_set_p as needed to achieve this number of gene sets
parser.add_option("","--filter-gene-set-metric-z",type=float,default=2.5) #gene sets with combined outlier metric z-score above this threshold are never seen (must have correct-huge turned on for this to work)
parser.add_option("","--max-gene-set-read-p",type=float,default=.05) #gene sets with p above this are excluded from the original beta analysis but included in gibbs
parser.add_option("","--min-gene-set-read-beta",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene stats file
parser.add_option("","--min-gene-set-read-beta-uncorrected",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene set stats file
parser.add_option("","--x-sparsify",type="string",action="callback",callback=get_comma_separated_args,default=[50,100,250,1000]) #applies to continuous gene sets, which are converted to dichotomous gene sets internally. For each value N, generate a new dichotomous gene set with the most N extreme genes (see next three options)
parser.add_option("","--add-ext",default=False,action="store_true") #add the top and bottom extremes as a gene set
parser.add_option("","--no-add-top",default=True,action="store_false",dest="add_top") #add the top extremes as a gene set
parser.add_option("","--no-add-bottom",default=True,action="store_false",dest="add_bottom") #add the bottom extremes as a gene set

parser.add_option("","--threshold-weights",type='float',default=0.5) #weights below this fraction of top weight are set to 0
parser.add_option("","--no-cap-weights",default=True,action="store_false",dest="cap_weights") #after normalizing weights by dividing by average, don't set those above 1 to have value 1
parser.add_option("","--max-gene-set-size",type=int,default=30000) #maximum number of genes in a gene set to consider
parser.add_option("","--add-all-genes",default=False,action="store_true") #add all genes from any gene set to the model, as opposed to just genes in the input --gwas-in or --exomes-in etc. Recommended to not normally use, since gene sets often are contaminated with genes that will bias toward significant associations. However, if you are passing in gene-values for only a small number of genes, and implicitly assuming that the remaining genes are zero, this can be used as a convenience feature rather than adding 0s for the desired genes
parser.add_option("","--prune-gene-sets",type=float,default=None) #gene sets with correlation above this threshold with any other gene set are removed (smallest gene set in correlation is retained)
parser.add_option("","--prune-deterministically",action="store_true") #prune in order of gene set size, not in order of p-value


#parameters for learning sigma
parser.add_option("","--chisq-dynamic",action="store_true") #dynamically determine the chisq threshold based on intercept and sigma
parser.add_option("","--desired-intercept-difference",type=float,default=1.3) #if dynamically determining chisq threshold, stop when intercept is less than this far away from 1
parser.add_option("","--chisq-threshold",type="float",default=5) #threshold outlier gene sets during sigma computation


#gene percentile parameters
parser.add_option("","--gene-percentiles-top-posterior",type=float,default=0.99) #specify maximum posterior, used in inverse normal conversion of percentile to posterior
parser.add_option("","--gene-percentiles-higher-is-better",default=False,action='store_true')
#gene z parameters
parser.add_option("","--gene-zs-gws-threshold",type=float,default=None) #specify significance threshold for genes to use for mapping to gws-posterior
parser.add_option("","--gene-zs-gws-prob-true",type=float,default=None) #specify probability genes at the significance threshold are true associations
#gene Z parameters
parser.add_option("","--gene-zs-max-mean-posterior",type=float,default=None) #specify significance threshold for genes to use for mapping to max-mean-posterior

#huge exomes parametersa
parser.add_option("","--exomes-high-p",type=float,default=5e-2) #specify the larger p-threshold for which we will constrain posterior
parser.add_option("","--exomes-high-p-posterior",type=float,default=0.1) #specify the posterior at the larger p-threshold
parser.add_option("","--exomes-low-p",type=float,default=2.5e-6) #specify the smaller p-threshold for which we will constrain posterior 
parser.add_option("","--exomes-low-p-posterior",type=float,default=0.95) #specify the posterior at the smaller p-threshold

#huge gwas parametersa
parser.add_option("","--gwas-high-p",type=float,default=1e-2) #specify the larger p-threshold for which we will constrain posterior
parser.add_option("","--gwas-high-p-posterior",type=float,default=0.01) #specify the posterior at the larger p-threshold
parser.add_option("","--gwas-low-p",type=float,default=5e-8) #specify the smaller p-threshold for which we will constrain posterior 
parser.add_option("","--gwas-low-p-posterior",type=float,default=0.75) #specify the posterior at the smaller p-threshold
parser.add_option("","--gwas-detect-low-power",type=int,default=None) #scale --gwas-low-p automatically to have at least this number signals reaching it; set to 0 to disable this
parser.add_option("","--gwas-detect-high-power",type=int,default=None) #scale --gwas-low-p automatically to have no more than this number of signals reaching it; set to a very high number to disable
parser.add_option("","--gwas-detect-no-adjust-huge",action="store_false",dest="gwas_detect_adjust_huge",default=True) #by default, --gwas-detect-power will affect the direct support and the prior calculations; enable this to keep the original huge scores but adjust detection just for prior calculations
parser.add_option("","--learn-window",default=False,action='store_true') #learn the window function linking SNPs to genes based on empirical distances of SNPs to genes and the --closest-gene-prob
parser.add_option("","--min-var-posterior",type=float,default=0.01) #exclude all variants with posterior below this; this uses the default parameters before detect low power
parser.add_option("","--closest-gene-prob",type=float,default=0.7) #specify probability that closest gene is the causal gene
#these control how the probability of a SNP to gene link is scaled, independently of how many genes there are nearby
parser.add_option("","--no-scale-raw-closest-gene",default=True,action='store_false',dest="scale_raw_closest_gene") #scale_raw_closest_gene: set everything to have the closest gene as closest gene prob (shifting up or down as necessary) 
parser.add_option("","--cap-raw-closest-gene",default=False,action='store_true') #cap_raw_closest_gene: set everything to have probability no greater than closest gene prob (shifting down but not up)
parser.add_option("","--max-closest-gene-prob",type=float,default=0.9) #specify maximum probability that closest gene is the causal gene. This accounts for probability that gene might just lie very far from the window
parser.add_option("","--max-closest-gene-dist",type=float,default=2.5e5) #the maximum distance for which we will search for the closest gene
#these parameters control how all genes nearby a signal are scaled
parser.add_option("","--no-cap-region-posterior",default=True,action='store_false',dest="cap_region_posterior") #ensure that the sum of gene probabilities is no more than 1
parser.add_option("","--scale-region-posterior",default=False,action='store_true') #ensure that the sum of gene probabilities is always 1
parser.add_option("","--phantom-region-posterior",default=False,action='store_true') #if the sum of gene probabilities is less than 1, assign the rest to a "phantom" gene that always has prior=0.05. As priors change for the other genes, they will "eat up" some of the phantom gene's assigned probability
parser.add_option("","--allow-evidence-of-absence",default=False,action='store_true') #allow the posteriors of genes to decrease below the background if there is a lack of GWAS signals
parser.add_option("","--correct-betas-mean",default=None,action='store_true',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)
parser.add_option("","--no-correct-betas-mean",default=None,action='store_false',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)
parser.add_option("","--correct-betas-var",default=False,action='store_true',dest="correct_betas_var") #don't correct gene set variables (var Z) for confounding variables (which still may exist even if all genes are corrected)

parser.add_option("","--min-n-ratio",type=float,default=0.5) #ignore SNPs with sample size less than this ratio of the max
parser.add_option("","--max-clump-ld",type=float,default=0.5) #maximum ld threshold to use for clumping (when MAF is passed in)
parser.add_option("","--signal-window-size",type=float,default=250000) #window size to initially include variants in a signal
parser.add_option("","--signal-min-sep",type=float,default=100000) #extend the region until the distance to the last significant snp is greater than the signal_min_sep
parser.add_option("","--signal-max-logp-ratio",type=float,default=None) #ignore all variants that are this ratio below max in signal
parser.add_option("","--credible-set-span",type=float,default=25000) #if user specified credible sets, ignore all variants within this var of a variant in the credible set

#sampling parameters
parser.add_option("","--max-num-burn-in",type=int,default=None) #maximum number of burn initerations to run

#sparsity parameters
parser.add_option("","--sparse-solution",default=None,action="store_true",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--no-sparse-solution",default=None,action="store_false",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--sparse-frac-gibbs",default=0.01,type=float) #zero out betas with with values below this fraction of the top; within the gibbs loop
parser.add_option("","--sparse-max-gibbs",default=0.001,type=float) #zero out betas with with values below this value; within the gibbs loop. Applies whether or not sparse-solution is set
parser.add_option("","--sparse-frac-betas",default=None,type=float) #zero out betas with with values below this fraction of the top, within each beta_tilde->beta calculation (within gibbs and prior to it). Only applied if sparse-solution is set

#priors parameters
parser.add_option("","--adjust-priors",default=None,action='store_true',dest="adjust_priors") #do correct priors for the number of gene sets a gene is in")
parser.add_option("","--no-adjust-priors",default=None,action='store_false',dest="adjust_priors") #do not correct priors for the number of gene sets a gene is in")

#gibbs parameters
parser.add_option("","--no-update-huge-scores",default=True,action='store_false',dest="update_huge_scores") #do not use priors to update huge scores (by default, priors affect "competition" for signal by nearby genes")
parser.add_option("","--top-gene-prior",type=float,default=None) #specify the top prior we are expecting any of the genes to have (after all of the calculations)
parser.add_option("","--increase-hyper-if-betas-below",type=float,default=None) #increase p if gene sets aren't significant enough

#factor parameters
parser.add_option("","--lmm-auth-key",default=None,type=str) #pass authorization key to enable LLM cluster labelling
parser.add_option("","--max-num-factors",default=30,type=int) #maximum k for factorization
parser.add_option("","--phi",default=0.05,type=float) #phi prior on factorization. Higher values yield fewer factors.
parser.add_option("","--alpha0",default=10,type=float) #alpha prior on lambda k for factorization (larger makes more sparse)
parser.add_option("","--beta0",default=1,type=float) #beta prior on lambda k for factorization
parser.add_option("","--gene-set-filter-value",type=float,default=0.01) #choose value of filter for gene sets. Will use beta uncorrected if available, otherwise beta, otherwise no filter
parser.add_option("","--gene-filter-value",type=float,default=1) #choose value of filter for genes. Will use combined if available, then priors, then Y, then nothing. Used only when anchoring to a pheno(s) (or default)
parser.add_option("","--pheno-filter-value",type=float,default=1) #choose value of filter for phenos. Used only when anchoring to genes
parser.add_option("","--gene-set-pheno-filter-value",type=float,default=0.01) #choose value of filter for gene set anchoring
parser.add_option("","--no-transpose",action='store_true') #factor original X rather than tranpose
parser.add_option("","--min-lambda-threshold",type=float,default=1e-3) #remove factors with lambdak values below this threshold, or sum(gene loadings) below this threshold, or sum(gene set loadings) below this threshold

#options for controlling factoring behavior
#Factoring decomposes the gene set x gene matrix or gene set x phenotype matrix while weighting matrix entries specific to an "anchor". An "anchor" can be either a single or set of phenotypes, or a single or set of genes
#
#Options for phenotype-based anchoring of factoring
#1. By default, factoring will be performed across the gene set x gene matrix and weighted by the gene combined scores or the gene set beta scores. We refer to this as "single phenotype anchoring" since the weights are based on associations of the genes and gene sets with the phenotype.
#   These will be computed as in a normal PIGEAN run, using --gwas-in or --positive-controls-in or --exomes-in or --gene-stats-in etc.
#   To specify this behavior, simply run factor (or factor_naive to generate betas and combined scores using the incorrect but faster naive approach)
#   Matrix to be factored is (probability gene relevant to anchor phenotype) * (probability gene set contains gene) * (probability that gene set is relevant to anchor phenotype)
#   Gene anchor loadings are (probability gene relevant to anchor phenotype) * (probability pathway contains gene) and gene set anchor loadings (probability that gene set interrogates pathway) * (probability that pathway is relevant to anchor phenotype)
#   Factor relevance scores are probability factor is relevant to anchor phenotype
#2. A special case of number 1 is to factor an input gene list. In this case, the gene list is treated like a "phenotype" and gene/gene set scores are determined that predict membership in it. So even though it is a gene list, it is *unrelated* to the gene sets used to construct the matrix. Semantically, it is equivalent to single phenotype anchoring
#   This is a way to decompose the gene set into distinct mechanisms and then (potentially) project it onto more phenotypes
#   To run this, use the --positive-controls-in (which allows weighting of genes in the set) or the --positive-controls-list options
#3. You can project the results of the "single phenotype anchoring" onto other phenotypes for which gene phewas or gene set phewas results are available. This will create factor loadings for all phenotypes in the file
#   To run this, use the --gene-set-phewas-stats-in option or the --gene-phewas-stats-in options alongside the factor command. If both --gene-set-phewas-stats-in and --gene-phewas-stats-in are specified, --gene-phewas-stats-in will be ignored and --gene-set-phewas-in will be used.
#   The factor command must include a way to obtain betas (e.g. as in a normal PIGEAN run) or you will get different behavior.
#   The interpretation of phenotype anchor loadings file is (probability phenotype associated with pathway) under this setting
#4. You can run a "multiple phenotype anchoring" factoring in which case the factorization will maximize the similarity of the approximated matrix to an input tensor, which is the input gene set x gene matrix projected along a third dimension (of length equal to the number of anchor phenotypes). Each matrix slice is obtained by the gene set x gene matrix (with entries equal to gene set weights) multiplied by the gene probabilities and gene set probabilities for the corresponding anchor phenotype.
#   There will be multiple loadings for each gene, gene set, and factor per anchor phenotype (interpretable identically to the single phenotype anchoring case), each sharing a core indicator loading multiplied by the gene or gene set probability. All phenotypes will be automatically projected as well as a part of this
#   To specify this behavior, run by passing in both --gene-phewas-stats-in and --gene-set-phewas-stats-in alongside factor model. Any other arguments to compute betas will be ignored here. You must also specify --anchor-phenos to determine the subset of phenotypes in the --phewas-stats-in to anchor on
#5. If you want to factor the entire gene by gene set matrix across all phenotypes in the --phewas-stats-in files, you specify the same flags as in the multiple phenotype anchoring case but replace --anchor-phenos with --anchor-any-pheno. This will not actually compute loadings for each phenotype, but rather will construct an "uber" phenotype that represents the probability that the gene or gene set is associated with any of the phenotypes.
#   The interpretation is as above, but instead of terms for (probability that X is relevant to the anchor phenotype) there are terms for (probability that X is relevant to any of the phenotypes)
#   You can use the flags --factor-prune-phenos-num or --factor-prune-phenos-val to reduce the number of phenotypes going into this analysis. --factor-prune-genes and --factor-prune-gene-sets are also important for limiting run time
#
#Options for gene-based anchoring of factoring
#6. To factor a gene set x phenotype matrix, you must anchor to a gene to determine the phenotype relevance scores and which gene sets to include. This is called "single gene anchoring"
#   You specify this behavior by passing in --gene-phewas-stats-in and --gene-set-phewas-stats-in (any flags for computing betas will be ignored) and then specifying --anchor-genes.
#   The entries in the input matrix represent (probability that gene is associated with phenotype) * (probability that gene set is associated with phenotype)
#   The anchor loadings files are, for phenotypes the (probability that the anchor gene is associated with phenotype) * (probability that phenotype is associated with the pathway) and, for gene sets, the (probability that the gene set interrogates the pathway)
#7. You can pass multiple comma separated values to --anchor-genes to run "multiple gene anchoring" factoring which behaves analogously to the mulitple phenotype anchoring. In this case, you can choose how gene sets are included in the matrix (either --add-gene-sets-by-enrichment-p, --add-gene-sets-by-fraction, --add-gene-sets-by-naive, or --add-gene-sets-by-gibbs)
#8. Finally, you can anchor across all genes using --anchor-any-gene. Just as for --anchor-any-pheno, this doesn't actually produce loadings for every anchor gene but instead uses weights for phenotypes corresponding to the probability that they are associated with any gene (these will usually be very close to 1)
#   To reduce the size of the matrix going into the factoring, you can use --factor-prune-phenos-num or --factor-prune-phenos-val which will remove phenotypes just as in the case of --anchor-any-pheno
#
#Options for gene-set-based anchoring of factoring
#9. You can factor the gene set x phenotype matrix if you specify --anchor-gene-set and pass in enough information to run a pheWAS
#   You need to pass in either positive controls, a GWAS, exomes, or gene-bfs, and then specify --run-phewas-from-gene-phewas-stats-in. This will then load the phewas statistics, which will then be used as weights in the factoring.
#   This will produce a single weight for the entire gene set, which is distinct from --anchor-gene [gene set] which will produce weights for each gene in the gene set as part of the factoring.
#   The entries in the input matrix represent (probability input gene set is associated with phenotype) * (probability that the input gene set is associated with gene set).
#   The anchor loadings will be (probability input gene set is associated with phenotype) * (probability that the phenotype is associated with the pathway) and (probability that the gene set interrogates the gene) * (probability that the gene set is associated with the input gene set)

#Note that the signle pheno anchoring with a gene list (option 2) and the multiple gene anchoring with the same gene list (option 7) both take a gene set as input, produce factors, and loadings across genes, phenos, and gene sets. The difference is in the interpretation of the factors. In the former case (option 2), the pathways are chosen to explain why each gene is in each gene set, with genes and gene sets weighted by similarity to the gene list. In the latter case (option 7), the pathways are chosen to explain why each pathway is associated with each phenotype, with phenotypes weighted according to how associated they are with each gene in the gene list


parser.add_option("","--gene-set-phewas-stats-in",default=None)
parser.add_option("","--gene-set-phewas-stats-id-col",default="Gene_Set")
parser.add_option("","--gene-set-phewas-stats-beta-col",default=None)
parser.add_option("","--gene-set-phewas-stats-beta-uncorrected-col",default=None)
parser.add_option("","--gene-set-phewas-stats-pheno-col",default=None)

parser.add_option("","--gene-phewas-bfs-in",default=None)
parser.add_option("","--gene-phewas-stats-in",dest="gene_phewas_bfs_in",default=None)
parser.add_option("","--gene-phewas-bfs-id-col",default=None)
parser.add_option("","--gene-phewas-stats-id-col",default=None,dest="gene_phewas_bfs_id_col")
parser.add_option("","--gene-phewas-bfs-log-bf-col",default=None)
parser.add_option("","--gene-phewas-stats-log-bf-col",default=None,dest="gene_phewas_bfs_log_bf_col")
parser.add_option("","--gene-phewas-bfs-combined-col",default=None)
parser.add_option("","--gene-phewas-stats-combined-col",default=None,dest="gene_phewas_bfs_combined_col")
parser.add_option("","--gene-phewas-bfs-prior-col",default=None)
parser.add_option("","--gene-phewas-stats-prior-col",default=None,dest="gene_phewas_bfs_prior_col")
parser.add_option("","--gene-phewas-bfs-pheno-col",default=None)
parser.add_option("","--gene-phewas-stats-pheno-col",default=None,dest="gene_phewas_bfs_pheno_col")
parser.add_option("","--min-gene-phewas-read-value",type="float",default=1)

parser.add_option("","--anchor-phenos",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None) #run single or multiple pheno anchoring
parser.add_option("","--anchor-pheno",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None,dest="anchor_phenos") #run single or multiple pheno anchoring
parser.add_option("","--anchor-any-pheno",action="store_true",default=False) #flatten all phenotypes into an uber weight
parser.add_option("","--anchor-genes",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None) #run single or multiple gene anchoring
parser.add_option("","--anchor-gene",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None,dest="anchor_genes") #run single or multiple gene anchoring
parser.add_option("","--anchor-any-gene",action="store_true",default=False) #update phenotype associations to essentially be uniformly 1
parser.add_option("","--anchor-gene-set",action="store_true",default=False) #run gene set anchoring

parser.add_option("","--factor-prune-phenos-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce phenotypes by including only this many (add an independent set). Phenotypes will be sorted by average probability across genes
parser.add_option("","--factor-prune-phenos-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce phenotypes by pruning those more correlated than this value. Phenotypes will be sorted by average probability across genes
parser.add_option("","--factor-prune-genes-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce genes by including only this many (add an independent set). Genes will be sorted by average probability across phenotypes
parser.add_option("","--factor-prune-genes-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce genes by pruning those more correlated than this value. Genes will be sorted by average probability across phenotypes
parser.add_option("","--factor-prune-gene-sets-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce gene sets by including only this many (add an independent set). Gene sets will be sorted by maximum association across phenotypes
parser.add_option("","--factor-prune-gene-sets-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce gene sets by pruning those more correlated than this value. Gene sets will be sorted by maximum assoication across phenotypes


parser.add_option("","--add-gene-sets-by-enrichment-p",type='float',default=None) #when running multiple gene anchoring, add in gene sets that pass the enrichment filters. Filter according to p-value
parser.add_option("","--add-gene-sets-by-fraction",type="float",default=None) #when running multiple gene anchoring, add in gene sets that have this fraction of input genes
parser.add_option("","--add-gene-sets-by-naive",type="float",default=None) #when running multiple gene anchoring, add in gene sets with beta_uncorrected above this threshold after naive
parser.add_option("","--add-gene-sets-by-gibbs",type="float",default=None) #when running multiple gene anchoring, add in gene sets with beta_uncorrected above this threshold after gibbs

#simulation parameters
parser.add_option("","--sim-log-bf-noise-sigma-mult",type=float,default=0) #noise to add to simulations (in standard devs)

#gibbs sampling parameters
parser.add_option("","--num-mad",type=int,default=10) #number of median absolute devs above which to treat chains as outliers
parser.add_option("","--min-num-iter",type=int,default=10) #minimum number of iterations to run for gibbs loop
parser.add_option("","--max-num-iter",type=int,default=500) #maximum number of iterations to run for gibbs loop
parser.add_option("","--num-chains",type=int,default=10) #number of chains for gibbs sampling
parser.add_option("","--r-threshold-burn-in",type=float,default=1.01) #maximum number of iterations to run for gibbs
parser.add_option("","--gauss-seidel",action="store_true") #run gauss seidel for gibbs sampling
parser.add_option("","--max-frac-sem",type=float,default=0.01) #the minimum z score (mean/sem) to allow after stopping sampling; continue sampling if this is too large
parser.add_option("","--use-sampled-betas-in-gibbs",action="store_true") #use a sample of the betas returned from the inner beta sampling within the gibbs samples; by default uses mean value which is smoother (more stable but more prone to not exploring full space)
parser.add_option("","--use-max-r-for-convergence",action="store_true") #use only the maximum R to evaluate convergence (most conservative). By default uses mean R

#beta sampling parameters
parser.add_option("","--min-num-iter-betas",type=int,default=10) #minimum number of iterations to run for beta sampling
parser.add_option("","--max-num-iter-betas",type=int,default=1100) #maximum number of iterations to run for beta sampling
parser.add_option("","--num-chains-betas",type=int,default=5) #number of chaings for beta sampling
parser.add_option("","--r-threshold-burn-in-betas",type=float,default=1.01) #threshold for R to consider a gene set as converged (that is, stop burn in and start sampling)
parser.add_option("","--gauss-seidel-betas",action="store_true") #run gauss seidel
parser.add_option("","--max-frac-sem-betas",type=float,default=0.01) #the minimum z score (mean/sem) to allow after stopping sampling; continue sampling if this is too large
parser.add_option("","--use-max-r-for-convergence-betas",action="store_true") #use only the maximum R across gene sets to evaluate convergence (most conservative). By default uses mean R


#TEMP DEBUGGING FLAGS
parser.add_option("","--debug-skip-phewas-covs",action="store_true") #
parser.add_option("","--debug-skip-huber",action="store_true") #
parser.add_option("","--debug-skip-correlation",action="store_true") #
parser.add_option("","--debug-zero-sparse",action="store_true") #
parser.add_option("","--debug-just-check-header",action="store_true") #
parser.add_option("","--debug-only-avg-huge",action="store_true")

(options, args) = parser.parse_args()

log_fh = None
if options.log_file is not None:
    log_fh = open(options.log_file, 'w')
else:
    log_fh = sys.stderr

NONE=0
INFO=1
DEBUG=2
TRACE=3
debug_level = options.debug_level
if debug_level is None:
    debug_level = INFO
def log(message, level=INFO, end_char='\n'):
    if level <= debug_level:
        log_fh.write("%s%s" % (message, end_char))
        log_fh.flush()

#set up warnings
warnings_fh = None
if options.warnings_file is not None:
    warnings_fh = open(options.warnings_file, 'w')
else:
    warnings_fh = sys.stderr

def warn(message):
    if warnings_fh is not None:
        warnings_fh.write("Warning: %s\n" % message)
        warnings_fh.flush()
    log(message, level=INFO)


try:
    options.x_sparsify = [int(x) for x in options.x_sparsify]
except ValueError:
    bail("option --x-sparsify: invalid integer list %s" % options.x_sparsify)

if len(args) < 1:
    bail(usage)

mode = args[0]

(options2, args2) = parser.parse_args()

run_huge = False
run_beta_tilde = False
run_sigma = False
run_beta = False
run_priors = False
run_naive_priors = False
run_gibbs = False
run_factor = False
run_phewas = False

run_naive_factor = False
run_sim = False
pops_defaults = False
use_phewas_for_factoring = False
factor_gene_set_x_pheno = False
expand_gene_sets = False


if mode == "huge" or mode == "huge_calc":
    run_huge = True
elif mode == "beta_tildes" or mode == "beta_tilde":
    run_beta_tilde = True
elif mode == "sigma":
    run_sigma = True
elif mode == "betas" or mode == "beta":
    run_beta = True
elif mode == "priors" or mode == "prior":
    run_priors = True
elif mode == "naive_priors" or mode == "naive_prior":
    run_naive_priors = True
elif mode == "gibbs" or mode == "em":
    run_gibbs = True
elif mode == "factor" or mode == "naive_factor": #run factoring, phewas factoring, or pheno factoring
    run_factor = True
    if options.add_gene_sets_by_naive is not None:
        run_naive_factor = True
    if mode == "naive_factor":
        run_naive_factor = True

    error = None
    if options.anchor_genes is not None and len(options.anchor_genes) == 1:
        factor_type = "single gene anchoring (to %s)" % options.anchor_genes
        if options.gene_set_phewas_stats_in is None or options.gene_phewas_bfs_in is None:
            error = "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    elif options.anchor_genes is not None and len(options.anchor_genes) > 1:
        factor_type = "multiple gene anchoring (to %s)" % options.anchor_genes
        if options.gene_set_phewas_stats_in is None or options.gene_phewas_bfs_in is None:
            error = "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    elif options.anchor_any_gene:
        factor_type = "any gene anchoring"
        if options.gene_set_phewas_stats_in is None or options.gene_phewas_bfs_in is None:
            error = "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    elif options.anchor_gene_set:
        factor_type = "gene set anchoring (to input phenotype/gene set)"
        if options.run_phewas_from_gene_phewas_stats_in is None:
            error = "Require --run-phewas-from-gene-phewas-stats"
    elif options.anchor_phenos is not None and len(options.anchor_phenos) == 1:
        factor_type = "single phenotype anchoring (to %s) but with phewas statistics used" % options.anchor_phenos
        if options.gene_set_phewas_stats_in is None or options.gene_phewas_bfs_in is None:
            error = "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    elif options.anchor_phenos is not None and len(options.anchor_phenos) > 1:
        factor_type = "multiple phenotype anchoring (to %s)" % options.anchor_phenos
        if options.gene_set_phewas_stats_in is None or options.gene_phewas_bfs_in is None:
            error = "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    elif options.anchor_any_pheno:
        factor_type = "any phenotype anchoring"
        if options.gene_set_phewas_stats_in is None or options.gene_phewas_bfs_in is None:
            error = "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    else:
        factor_type = "single phenotype anchoring (to %s) using default statistics" % options.anchor_phenos
        if options.gene_set_phewas_stats_in is not None or options.gene_phewas_bfs_in is not None:
            factor_type = "%s. Will project using %s" % (factor_type, options.gene_set_phewas_stats_in if options.gene_set_phewas_stats_in is not None else options.gene_phewas_bfs_in)

    factor_gene_set_x_pheno = options.anchor_genes or options.anchor_any_gene or options.anchor_gene_set
    use_phewas_for_factoring = options.anchor_phenos is not None or options.anchor_any_pheno or options.anchor_genes is not None or options.anchor_any_gene
    expand_gene_sets = options.anchor_genes is not None and len(options.anchor_genes) > 1

    if (options.add_gene_sets_by_enrichment_p is not None or options.add_gene_sets_by_fraction is not None or options.add_gene_sets_by_naive is not None or options.add_gene_sets_by_gibbs is not None) and not expand_gene_sets:
        warn("Ignoring options to add gene sets based on association with anchor genes because only 1 anchor gene was specified")

    if error is not None:
        bail("Cannot run factoring type: %s. %s" % (factor_type, error))
    else:
        log("Running factoring type: %s" % factor_type)

    if ((use_phewas_for_factoring or factor_gene_set_x_pheno) and not options.anchor_gene_set) and (options.gene_set_stats_in or options.gene_zs_in or options.gene_percentiles_in or options.gwas_in or options.exomes_in or options.positive_controls_in or options.positive_controls_list is not None):
        if use_phewas_for_factoring:
            warn("Ignoring all arguments for reading Y or reading betas in --anchor-phenos mode")
        elif factor_gene_set_x_pheno:
            warn("Ignoring all arguments for reading Y or reading betas in --anchor-genes mode")

elif mode == "sim" or mode == "simulate":
    run_sim = True
elif mode == "pops":
    pops_defaults = True
    run_priors = True
elif mode == "naive_pops":
    pops_defaults = True
    run_naive_priors = True
else:
    bail("Unrecognized mode %s" % mode)

if options.run_phewas_from_gene_phewas_stats_in is not None:
    run_phewas = True

#set defaults

if mode == "pops" or mode == "naive_pops":

    options.correct_betas_mean = options.correct_betas_mean if options.correct_betas_mean is not None else False
    options.adjust_priors = options.adjust_priors if options.adjust_priors is not None else False
    options.p_noninf = options.p_noninf if options.p_noninf is not None else 1
    options.sigma_power = options.sigma_power if options.sigma_power is not None else 2
    options.update_hyper = options.update_hyper if options.update_hyper is not None else "none"
    options.filter_negative = options.filter_negative if options.filter_negative is not None else False
    options.prune_gene_sets = options.prune_gene_sets if options.prune_gene_sets is not None else 1.1
    options.top_gene_set_prior = options.top_gene_set_prior if options.top_gene_set_prior is not None else 0.1
    options.num_gene_sets_for_prior = options.num_gene_sets_for_prior if options.num_gene_sets_for_prior is not None else 15000
    options.filter_gene_set_p = options.filter_gene_set_p if options.filter_gene_set_p is not None else 0.05
    options.linear = options.linear if options.linear is not None else True
    options.max_for_linear = options.max_for_linear if options.max_for_linear is not None else 1
    options.min_gene_set_size = options.min_gene_set_size if options.min_gene_set_size is not None else 1
    options.cross_val = options.cross_val if options.cross_val is not None else True
    options.sparse_frac_betas = options.sparse_frac_betas if options.sparse_frac_betas is not None else 0
    options.sparse_solution = options.sparse_solution if options.sparse_solution is not None else False

    options.gene_zs_id_col = options.gene_zs_id_col if options.gene_zs_id_col is not None else "GENE"
    options.gene_zs_value_col = options.gene_zs_value_col if options.gene_zs_value_col is not None else "ZSTAT"
else:
    options.correct_betas_mean = options.correct_betas_mean if options.correct_betas_mean is not None else True
    options.adjust_priors = options.adjust_priors if options.adjust_priors is not None else True
    options.p_noninf = options.p_noninf if options.p_noninf is not None else 0.001
    options.sigma_power = options.sigma_power if options.sigma_power is not None else -2
    options.update_hyper = options.update_hyper if options.update_hyper is not None else "p"
    options.filter_negative = options.filter_negative if options.filter_negative is not None else True
    if options.prune_gene_sets is None:
        if run_factor and factor_gene_set_x_pheno is not None:
            options.prune_gene_sets = 0.5
        else:
            options.prune_gene_sets = 0.8

    options.top_gene_set_prior = options.top_gene_set_prior if options.top_gene_set_prior is not None else 0.8
    options.num_gene_sets_for_prior = options.num_gene_sets_for_prior if options.num_gene_sets_for_prior is not None else 50
    options.filter_gene_set_p = options.filter_gene_set_p if options.filter_gene_set_p is not None else 0.01
    options.linear = options.linear if options.linear is not None else False
    options.max_for_linear = options.max_for_linear if options.max_for_linear is not None else 0.95
    options.min_gene_set_size = options.min_gene_set_size if options.min_gene_set_size is not None else 1

    if run_factor and factor_gene_set_x_pheno is not None:
        if options.add_gene_sets_by_enrichment_p is not None:
            options.filter_gene_set_p = options.add_gene_sets_by_enrichment_p

    options.cross_val = options.cross_val if options.cross_val is not None else False
    options.sparse_frac_betas = options.sparse_frac_betas if options.sparse_frac_betas is not None else 0.001
    options.sparse_solution = options.sparse_solution if options.sparse_solution is not None else True

if options.gene_cor_file is None and options.gene_loc_file is None and not options.ols:
    warn("Switching to run --ols since --gene-cor-file and --gene-loc-file are unspecified")
    options.ols = True


def urlopen_with_retry(file, flag=None, tries=5, delay=60, backoff=2):
    import urllib.request
    import urllib.error

    while tries > 1:
        try:
            if flag is not None:
                return urllib.request.urlopen(file, flag)
            else:
                return urllib.request.urlopen(file)
        except urllib.error.URLError as e:
            log("%s, Retrying in %d seconds..." % (str(e), delay))
            time.sleep(delay)
            tries -= 1
            delay *= backoff
    bail("Couldn't open file after too many retries")


def is_gz_file(filepath, is_remote, flag=None):

    if len(filepath) >= 3 and (filepath[-3:] == ".gz" or filepath[-4:] == ".bgz") and (flag is None or 'w' not in flag):
        try:
            if is_remote:
                test_fh = urlopen_with_retry(filepath)
            else:
                test_fh = gzip.open(filepath, 'rb')

            try:
                test_fh.readline()
                test_fh.close()
                return True
            except Exception:
                return False

        except FileNotFoundError:
            return True

    elif flag is None or 'w' not in flag:
        flag = 'rb'
        if is_remote:
            test_fh = urlopen_with_retry(filepath)
        else:
            test_fh = open(filepath, 'rb')

        is_gz = test_fh.read(2) == b'\x1f\x8b'
        test_fh.close()
        return is_gz
    else:
        return filepath[-3:] == ".gz" or filepath[-4:] == ".bgz"

def open_gz(file, flag=None):
    is_remote = False
    remote_prefixes = ["http:", "https:", "ftp:"]
    for remote_prefix in remote_prefixes:
        if len(file) >= len(remote_prefix) and file[:len(remote_prefix)] == remote_prefix:
            is_remote = True

    if is_gz_file(file, is_remote, flag=flag):
        open_fun = gzip.open
        if flag is not None and len(flag) > 0 and not flag[-1] == 't':
            flag = "%st" % flag
        elif flag is None:
            flag = "rt"
    else:
        open_fun = open

    if is_remote:
        import io
        if flag is not None:
            if open_fun is open:
                fh = io.TextIOWrapper(urlopen_with_retry(file, flag))
            else:
                fh = open_fun(urlopen_with_retry(file), flag)
        else:
            if open_fun is open:
                fh = io.TextIOWrapper(urlopen_with_retry(file))
            else:
                fh = open_fun(urlopen_with_retry(file))
    else:
        if flag is not None:
            try:
                fh = open_fun(file, flag, encoding="utf-8")
            except LookupError:
                fh = open_fun(file, flag)
        else:
            try:
                fh = open_fun(file, encoding="utf-8")
            except LookupError:
                fh = open_fun(file)

    return fh

class GeneSetData(object):
    '''
    Stores gene and gene set annotations and derived matrices
    It allows reading X or V files and using these to determine the allowed gene sets and genes
    '''
    def __init__(self, background_prior=0.05, batch_size=4500):

        #empirical mean scale factor from mice
        self.MEAN_MOUSE_SCALE = 0.0448373

        if background_prior <= 0 or background_prior >= 1:
            bail("--background-prior must be in (0,1)")
        self.background_prior = background_prior
        self.background_log_bf = np.log(self.background_prior / (1 - self.background_prior))
        self.background_bf = np.exp(self.background_log_bf)


        #genes x gene set indicator matrix (sparse)
        #this is always the original matrix -- it is never rescaled or shifted
        #but, calculations of beta_tildes etc. are done relative to what would be obtained if it were scaled
        #similarly, when X/y is whitened, the "internal state" of the code is that both X and y are whitened (and scale factors reflect this)
        #but, for efficiency, X is maintained as a sparse matrix
        #so, ideally this should never be accessed directly; instead get_X_orig returns the original (sparse) matrix and makes the intent explicity to avoid any scaling/whitening
        #get_X_blocks returns the (unscaled) but whitened X
        self.X_orig = None
        #these are genes that we want to calculate priors for but which don't have gene-level statistics
        self.X_orig_missing_genes = None
        self.X_orig_missing_genes_missing_gene_sets = None
        self.X_orig_missing_gene_sets = None
        #internal cache
        self.last_X_block = None

        #genes x gene set normalized matrix
        #REMOVING THIS for memory savings
        #self.X = None

        #this is the number of gene sets to put into a batch when fetching blocks of X
        self.batch_size = batch_size

        #flag to indicate whether these scale factors correspond to X_orig or the (implicit) whitened version
        #if True, they can be used directly with _get_X_blocks
        #if False but y_corr_cholesky is True, then they need to be recomputed
        self.scale_is_for_whitened = False
        self.scale_factors = None
        self.mean_shifts = None

        self.scale_factors_missing = None
        self.mean_shifts_missing = None

        self.scale_factors_ignored = None
        self.mean_shifts_ignored = None

        #whether this was originally a dense or sparse gene set
        self.is_dense_gene_set = None
        self.is_dense_gene_set_missing = None

        self.gene_set_batches = None
        self.gene_set_batches_missing = None

        self.gene_set_labels = None
        self.gene_set_labels_missing = None
        self.gene_set_labels_ignored = None

        #ordered list of genes
        self.genes = None
        self.genes_missing = None
        self.gene_to_ind = None
        self.gene_missing_to_ind = None

        self.gene_chrom_name_pos = None
        self.gene_to_chrom = None
        self.gene_to_pos = None
        self.gene_to_gwas_huge_score = None
        self.gene_to_gwas_huge_score_uncorrected = None
        self.gene_to_exomes_huge_score = None
        self.gene_to_huge_score = None

        self.anchor_pheno_mask = None
        self.anchor_gene_mask = None

        self.default_pheno_mask = None

        self.gene_pheno_combined_prior_Ys = None
        self.gene_pheno_Y = None
        self.gene_pheno_priors = None

        self.num_gene_phewas_filtered = 0

        #note that these phewas betas are all stored in *external* units (by contrast to the betas which are in internal units)

        #these are for phewas. Regression of pheno values (combined, Y, or priors) against input Y values

        self.pheno_Y_vs_input_Y_beta = None
        self.pheno_Y_vs_input_Y_beta_tilde = None
        self.pheno_Y_vs_input_Y_se = None
        self.pheno_Y_vs_input_Y_Z = None
        self.pheno_Y_vs_input_Y_p_value = None

        self.pheno_combined_prior_Ys_vs_input_Y_beta = None
        self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde = None
        self.pheno_combined_prior_Ys_vs_input_Y_se = None
        self.pheno_combined_prior_Ys_vs_input_Y_Z = None
        self.pheno_combined_prior_Ys_vs_input_Y_p_value = None

        #these are for phewas. Regression of pheno values (combined, Y, or priors) against input combined values
        self.pheno_Y_vs_input_combined_prior_Ys_beta = None
        self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde = None
        self.pheno_Y_vs_input_combined_prior_Ys_se = None
        self.pheno_Y_vs_input_combined_prior_Ys_Z = None
        self.pheno_Y_vs_input_combined_prior_Ys_p_value = None

        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z = None
        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value = None

        #these are for phewas. Regression of pheno values (combined, Y, or priors) against input prior values
        self.pheno_Y_vs_input_priors_beta = None
        self.pheno_Y_vs_input_priors_beta_tilde = None
        self.pheno_Y_vs_input_priors_se = None
        self.pheno_Y_vs_input_priors_Z = None
        self.pheno_Y_vs_input_priors_p_value = None

        self.pheno_combined_prior_Ys_vs_input_priors_beta = None
        self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde = None
        self.pheno_combined_prior_Ys_vs_input_priors_se = None
        self.pheno_combined_prior_Ys_vs_input_priors_Z = None
        self.pheno_combined_prior_Ys_vs_input_priors_p_value = None

        self.gene_to_positive_controls = None

        self.gene_label_map = None

        #only used for running factoring by phenotype
        self.phenos = None
        self.pheno_to_ind = None

        #note that these phewas betas are all stored in *external* units (by contrast to the betas which are in internal units)
        self.X_phewas_beta_uncorrected = None
        self.X_phewas_beta = None

        #ordered list of gene sets
        self.gene_sets = None
        self.gene_sets_missing = None
        self.gene_sets_ignored = None
        self.gene_set_to_ind = None

        #gene set association statistics
        #self.max_gene_set_p = None

        #self.is_logistic = None

        self.beta_tildes = None
        self.p_values = None
        self.ses = None
        self.z_scores = None

        self.beta_tildes_orig = None
        self.p_values_orig = None
        self.ses_orig = None
        self.z_scores_orig = None

        #these store the inflation of SE relative to OLS (if ols_corrected is run)
        self.se_inflation_factors = None

        #these are gene sets we filtered out but need to persist for OSC
        self.beta_tildes_missing = None
        self.p_values_missing = None
        self.ses_missing = None
        self.z_scores_missing = None
        self.se_inflation_factors_missing = None

        #these are gene sets we ignored at the start
        self.col_sums_ignored = None

        self.beta_tildes_ignored = None
        self.p_values_ignored = None
        self.ses_ignored = None
        self.z_scores_ignored = None
        self.se_inflation_factors_ignored = None

        self.beta_tildes_missing_orig = None
        self.p_values_missing_orig = None
        self.ses_missing_orig = None
        self.z_scores_missing_orig = None

        #DO WE NEED THIS???
        #self.y_mean = None
        self.Y = None
        self.Y_exomes = None
        self.Y_positive_controls = None

        #this is to store altered variables if we detect power
        #these are used for fitting the betas (the indirect support)
        #self.Y is the direct support and is used only for combining with the indirect support to get a D value for this gene
        self.Y_for_regression = None

        #this is where to store the original uncorrected Y values if we have them
        self.Y_uncorrected = None

        self.y_var = 1 #total variance of the Y
        self.Y_orig = None
        self.Y_for_regression_orig = None
        self.Y_w_orig = None
        self.Y_fw_orig = None


        self.gene_locations = None #this stores sort orders for genes, which is populated when fitting correlation matrix from gene loc file

        self.huge_signal_bfs = None
        self.huge_signal_bfs_for_regression = None

        #covariates for genes
        self.gene_covariates = None
        self.gene_covariates_mask = None
        self.gene_covariate_names = None
        self.gene_covariate_directions = None
        self.gene_covariate_intercept_index = None
        self.gene_covariates_mat_inv = None
        self.gene_covariate_zs = None
        self.gene_covariate_adjustments = None

        #for sparse mode
        self.huge_sparse_mode = False
        self.gene_covariate_slope_defaults = None
        self.total_qc_metric_betas_defaults = None
        self.total_qc_metric_intercept_defaults = None
        self.total_qc_metric2_betas_defaults = None
        self.total_qc_metric2_intercept_defaults = None


        self.total_qc_metric_betas = None
        self.total_qc_metric_intercept = None
        self.total_qc_metric2_betas = None
        self.total_qc_metric2_intercept = None
        self.total_qc_metric_desired_var = None

        self.huge_signals = None
        self.huge_signal_posteriors = None
        self.huge_signal_posteriors_for_regression = None
        self.huge_signal_sum_gene_cond_probabilities = None
        self.huge_signal_sum_gene_cond_probabilities_for_regression = None
        self.huge_signal_mean_gene_pos = None
        self.huge_signal_mean_gene_pos_for_regression = None        
        self.huge_signal_max_closest_gene_prob = None

        self.huge_cap_region_posterior = True
        self.huge_scale_region_posterior = False
        self.huge_phantom_region_posterior = False
        self.huge_allow_evidence_of_absence = False

        self.y_corr = None #this stores the (banded) correlation matrix for the Y values
        #In addition to storing banded correlation matrix, this signals that we are in partial GLS mode (OLS with inflated SEs)
        self.y_corr_sparse = None #another representation of the banded correlation matrix
        #In addition to storing cholesky decomp, this being set to not None triggers everything to operate in full GLS mode
        self.y_corr_cholesky = None #this stores the cholesky decomposition of the (banded) correlation matrix for the Y values
        #these are the "whitened" ys that are multiplied by sigma^{-1/2}
        self.Y_w = None
        self.y_w_var = 1 #total variance of the whitened Y
        self.y_w_mean = 0 #total mean of the whitened Y
        #these are the "full whitened" ys that are multiplied by sigma^{-1}
        self.Y_fw = None
        self.y_fw_var = 1 #total variance of the whitened Y
        self.y_fw_mean = 0 #total mean of the whitened Y

        #statistics for sigma regression
        self.osc = None
        self.X_osc = None
        self.osc_weights = None

        self.osc_missing = None
        self.X_osc_missing = None
        self.osc_weights_missing = None

        #statistics for gene set qc
        self.total_qc_metrics = None
        self.mean_qc_metrics = None

        self.total_qc_metrics_missing = None
        self.mean_qc_metrics_missing = None

        self.total_qc_metrics_ignored = None
        self.mean_qc_metrics_ignored = None

        self.total_qc_metrics_directions = None

        self.p = None
        self.ps = None #this allows gene sets to have different ps
        self.ps_missing = None #this allows gene sets to have different ps
        self.sigma2 = None #sigma2 * np.power(scale_factor, sigma_power) is the prior used for the internal beta
        self.sigma2s = None #this allows gene sets to have different sigma2s
        self.sigma2s_missing = None #this allows gene sets to have different sigma2s

        self.sigma2_osc = None
        self.sigma2_se = None
        self.intercept = None
        self.sigma2_p = None
        self.sigma2_total_var = None
        self.sigma2_total_var_lower = None
        self.sigma2_total_var_upper = None

        #statistics for gene set betas
        self.betas = None
        self.betas_uncorrected = None
        self.inf_betas = None
        self.non_inf_avg_cond_betas = None
        self.non_inf_avg_postps = None

        self.betas_missing = None
        self.betas_uncorrected_missing = None
        self.inf_betas_missing = None
        self.non_inf_avg_cond_betas_missing = None
        self.non_inf_avg_postps_missing = None

        self.betas_orig = None
        self.betas_uncorrected_orig = None
        self.inf_betas_orig = None
        self.non_inf_avg_cond_betas_orig = None
        self.non_inf_avg_postps_orig = None

        self.betas_missing_orig = None
        self.betas_uncorrected_missing_orig = None
        self.inf_betas_missing_orig = None
        self.non_inf_avg_cond_betas_missing_orig = None
        self.non_inf_avg_postps_missing_orig = None

        #statistics for genes
        self.priors = None
        self.priors_adj = None
        self.combined_prior_Ys = None
        self.combined_prior_Ys_for_regression = None

        self.combined_prior_Ys_adj = None
        self.combined_prior_Y_ses = None
        self.combined_Ds = None
        self.combined_Ds_for_regression = None
        self.combined_Ds_missing = None
        self.priors_missing = None
        self.priors_adj_missing = None

        self.gene_N = None
        self.gene_ignored_N = None #number of ignored gene sets gene is in

        self.gene_N_missing = None #gene_N for genes with missing values for Y
        self.gene_ignored_N_missing = None #gene_N_missing for genes with missing values for Y

        self.batches = None

        self.priors_orig = None
        self.priors_adj_orig = None
        self.priors_missing_orig = None
        self.priors_adj_missing_orig = None

        #model parameters
        self.sigma_power = None

        #soft thresholding of sigmas
        self.sigma_threshold_k = None
        self.sigma_threshold_xo = None

        #stores all parameters used
        self.params = {}
        self.param_keys = []

        #stores factored matrices
        self.exp_lambdak = None #anchor-agnostic factor relevance weights (does the factor exist)
        self.factor_anchor_relevance = None #relevance of each factor to each anchor
        self.factor_relevance = None #max relevance of each factor across anchors

        #these are specific to the anchor-agnostic loadings
        self.factor_labels = None
        self.factor_top_gene_sets = None
        self.factor_top_genes = None
        self.factor_top_phenos = None

        #these are specific to anchors
        self.factor_anchor_top_gene_sets = None
        self.factor_anchor_top_genes = None
        self.factor_anchor_top_phenos = None

        #masks used to select inputs to the factoring
        self.gene_factor_gene_mask = None
        self.gene_set_factor_gene_set_mask = None
        self.pheno_factor_pheno_mask = None  #only used in factor pheno mode or factor phewas mode

        self.exp_gene_factors = None #anchor-agnostic factor loadings
        self.gene_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings

        self.exp_gene_set_factors = None  #anchor-agnostic factor loadings
        self.gene_set_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings

        self.exp_pheno_factors = None #anchor-agnostic factor loadings
        self.pheno_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings

        self.factor_phewas_Y_betas = None #phewas statistics
        self.factor_phewas_Y_ses = None #phewas statistics
        self.factor_phewas_Y_zs = None #phewas statistics
        self.factor_phewas_Y_p_values = None #phewas statistics
        self.factor_phewas_Y_one_sided_p_values = None #phewas statistics

        self.factor_phewas_Y_huber_betas = None #phewas statistics
        self.factor_phewas_Y_huber_ses = None #phewas statistics
        self.factor_phewas_Y_huber_zs = None #phewas statistics
        self.factor_phewas_Y_huber_p_values = None #phewas statistics
        self.factor_phewas_Y_huber_one_sided_p_values = None #phewas statistics

        self.factor_phewas_combined_prior_Ys_betas = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_ses = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_zs = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_p_values = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_one_sided_p_values = None #phewas statistics

        self.factor_phewas_combined_prior_Ys_huber_betas = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_ses = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_zs = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_p_values = None #phewas statistics
        self.factor_phewas_combined_prior_Ys_huber_one_sided_p_values = None #phewas statistics

    def init_gene_locs(self, gene_loc_file):
        log("Reading --gene-loc-file %s" % gene_loc_file)
        (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

    def read_gene_map(self, gene_map_in, gene_map_orig_gene_col=1, gene_map_new_gene_col=2):

        if self.gene_label_map is None:
            self.gene_label_map = {}

        gene_map_orig_gene_col -= 1
        if gene_map_orig_gene_col < 0:
            bail("--gene-map-orig-gene-col must be greater than 1")
        gene_map_new_gene_col -= 1
        if gene_map_new_gene_col < 0:
            bail("--gene-map-new-gene-col must be greater than 1")

        with open(gene_map_in) as map_fh:
            for line in map_fh:
                cols = line.strip().split()
                if len(cols) <= gene_map_orig_gene_col or len(cols) <= gene_map_new_gene_col:
                    bail("Not enough columns in --gene-map-in:\n\t%s" % line)

                orig_gene = cols[0]
                new_gene = cols[1]
                self.gene_label_map[orig_gene] = new_gene

    def remove_genes_for_phewas_factor(self, keep_genes=None, naive_gt=None, num_gt=None, num_top_per_factor=None, num_top=None):
        if self.genes is None:
            return

        keep_genes_mask = np.full(len(self.genes), False)
        if keep_genes is not None:
            keep_genes_mask = np.array([x in keep_genes for x in self.genes])

        gene_mask = np.full(len(self.genes), True)

        if naive_gt is not None and self.priors is not None:
            gene_mask = np.logical_and(gene_mask, self.priors > naive_gt)
        if num_gt is not None and self.X_orig is not None:
            gene_mask = np.logical_and(gene_mask, self.X_orig.sum(axis=1).A1 > num_gt)

        if np.sum(gene_mask) != len(gene_mask):
            gene_mask = np.logical_or(gene_mask, keep_genes_mask)

            log("Removing %d genes (kept %d) due to requested filters" % (len(gene_mask) - np.sum(gene_mask), np.sum(gene_mask)), DEBUG)
            self._subset_genes(gene_mask, skip_V=True, overwrite_missing=True, skip_scale_factors=False)
            keep_genes_mask = keep_genes_mask[gene_mask]

        if num_top_per_factor is not None and self.exp_gene_factors is not None and self.exp_gene_factors.shape[1] > 0 and num_top_per_factor < len(self.genes):

            gene_mask = np.full(len(self.genes), False)
            num_bottom = len(self.genes) - num_top_per_factor

            highest_row_indices_per_column = np.argpartition(self.exp_gene_factors, num_bottom, axis=0)[-num_top_per_factor:,:]
            gene_mask[np.unique(highest_row_indices_per_column)] = True

            if np.sum(gene_mask) != len(gene_mask):
                gene_mask = np.logical_or(gene_mask, keep_genes_mask)
                log("Removing %d genes (kept %d) due to requested number of top per factor" % (len(gene_mask) - np.sum(gene_mask), np.sum(gene_mask)), DEBUG)
                self._subset_genes(gene_mask, skip_V=True, overwrite_missing=True, skip_scale_factors=False)
                keep_genes_mask = keep_genes_mask[gene_mask]

        if num_top is not None and num_top < len(self.genes):
            values_for_top = self.X_orig.sum(axis=1).A1
            if self.exp_gene_factors is not None:
                values_for_top = np.max(self.exp_gene_factors, axis=1)
            elif naive_gt is not None and self.priors is not None:
                values_for_top = self.priors

            gene_mask = np.full(len(self.genes), False)
            num_bottom = len(self.genes) - num_top

            gene_mask[np.argpartition(values_for_top, num_bottom)[-num_top:]] = True
            log("Removing %d genes (kept %d) due to requested number of top" % (len(gene_mask) - np.sum(gene_mask), np.sum(gene_mask)), DEBUG)
            gene_mask = np.logical_or(gene_mask, keep_genes_mask)
            self._subset_genes(gene_mask, skip_V=True, overwrite_missing=True, skip_scale_factors=False)
            keep_genes_mask = keep_genes_mask[gene_mask]


    def read_Y(self, gwas_in=None, exomes_in=None, positive_controls_in=None, positive_controls_list=None, gene_bfs_in=None, gene_percentiles_in=None, gene_zs_in=None, gene_loc_file=None, gene_covs_in=None, hold_out_chrom=None, **kwargs):

        Y1_exomes = np.array([])
        extra_genes_exomes = []
        extra_Y_exomes = []

        def __hold_out_chrom(Y, extra_genes, extra_Y):
            if hold_out_chrom is None:
                return (Y, extra_genes, extra_Y)

            if self.gene_to_chrom is None:
                (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

            extra_Y_mask = np.full(len(extra_Y), True)
            for i in range(len(extra_genes)):
                if extra_genes[i] in self.gene_to_chrom and self.gene_to_chrom[extra_genes[i]] == hold_out_chrom:
                    extra_Y_mask[i] = False
            if np.sum(~extra_Y_mask) > 0:
                extra_genes = [extra_genes[i] for i in range(len(extra_genes)) if extra_Y_mask[i]]
                extra_Y = extra_Y[extra_Y_mask]

            if self.genes is not None:
                Y_nan_mask = np.full(len(Y), False)
                for i in range(len(self.genes)):
                    if self.genes[i] in self.gene_to_chrom and self.gene_to_chrom[self.genes[i]] == hold_out_chrom:
                        Y_nan_mask[i] = True
                if np.sum(Y_nan_mask) > 0:
                    Y[Y_nan_mask] = np.nan

            return (Y, extra_genes, extra_Y)

        if exomes_in is not None:
            (Y1_exomes,extra_genes_exomes,extra_Y_exomes) = self.calculate_huge_scores_exomes(exomes_in, hold_out_chrom=hold_out_chrom, gene_loc_file=gene_loc_file, **kwargs)
            if self.genes is None:
                self._set_X(self.X_orig, extra_genes_exomes, self.gene_sets, skip_N=True, skip_V=True)
                #set this temporarily for use in huge
                self.Y_exomes = extra_Y_exomes
                Y1_exomes = extra_Y_exomes
                extra_genes_exomes = []
                extra_Y_exomes = np.array([])


        missing_value_exomes = 0
        missing_value_positive_controls = 0

        Y1_positive_controls = np.array([])
        extra_genes_exomes_positive_controls = extra_genes_exomes
        extra_Y_positive_controls = []

        if positive_controls_in is not None or positive_controls_list is not None:
            (Y1_positive_controls,extra_genes_positive_controls,extra_Y_positive_controls) = self.read_positive_controls(positive_controls_in, positive_controls_list=positive_controls_list, hold_out_chrom=hold_out_chrom, gene_loc_file=gene_loc_file, **kwargs)
            if self.genes is None:
                assert(len(Y1_exomes) == 0)
                self._set_X(self.X_orig, extra_genes_positive_controls, self.gene_sets, skip_N=True, skip_V=True)
                #set this temporarily for use in huge
                self.Y_positive_controls = extra_Y_positive_controls
                Y1_positive_controls = extra_Y_positive_controls
                extra_genes_positive_controls = []
                extra_Y_positive_controls = np.array([])
                Y1_exomes = np.zeros(len(Y1_positive_controls))
            else:
                #exomes is already aligned to self.genes: Y1_exomes matches self.genes
                #extra_genes_exomes / extra_Y_exomes has anything not in it
                #we need to:
                #1. make sure Y1_exomes and Y1_positive_controls are the same length (already done)
                #2. combine extra_genes for the two
                #3. make sure that the 


                #align these so that genes includes the union of exomes and positive controls
                #and the extras moved in are no longer in extra
                #only need to remove the extras from exomes, since it was loaded into self.genes already
                extra_gene_to_ind = self._construct_map_to_ind(extra_genes_positive_controls)
                extra_Y_positive_controls = list(extra_Y_positive_controls)
                new_extra_Y_exomes = list(np.full(len(extra_Y_positive_controls), missing_value_exomes))
                num_add = 0
                extra_genes_exomes_positive_controls = extra_genes_positive_controls
                for i in range(len(extra_genes_exomes)):
                    if extra_genes_exomes[i] in extra_gene_to_ind:
                        new_extra_Y_exomes[extra_gene_to_ind[extra_genes_exomes[i]]] = extra_Y_exomes[i]
                    else:
                        num_add += 1
                        extra_genes_exomes_positive_controls.append(extra_genes_exomes[i])
                        extra_Y_positive_controls.append(missing_value_positive_controls)
                        new_extra_Y_exomes.append(extra_Y_exomes[i])

                extra_Y_exomes = np.array(new_extra_Y_exomes)
                extra_Y_positive_controls = np.array(extra_Y_positive_controls)
        else:
            Y1_positive_controls = np.zeros(len(Y1_exomes))
            extra_Y_positive_controls = np.zeros(len(extra_Y_positive_controls))
            extra_genes_exomes_positive_controls = extra_genes_exomes

        assert(len(extra_Y_exomes) == len(extra_genes_exomes_positive_controls))
        assert(len(extra_Y_exomes) == len(extra_Y_positive_controls))
        assert(len(Y1_exomes) == len(Y1_positive_controls))

        missing_value = None
        gene_combined_map = None
        gene_prior_map = None

        #read these in here if there is a file

        if gwas_in is not None:
            (Y1,extra_genes,extra_Y,Y1_for_regression,extra_Y_for_regression) = self.calculate_huge_scores_gwas(gwas_in, gene_loc_file=gene_loc_file, hold_out_chrom=hold_out_chrom, **kwargs)
            missing_value = 0
        else:
            self.huge_signal_bfs = None
            self.huge_signal_bfs_for_regression = None

            if gene_bfs_in is not None:
                (Y1,extra_genes,extra_Y, gene_combined_map, gene_prior_map)  = self._read_gene_bfs(gene_bfs_in, **kwargs)
            elif gene_percentiles_in is not None:
                (Y1,extra_genes,extra_Y) = self._read_gene_percentiles(gene_percentiles_in, **kwargs)
            elif gene_zs_in is not None:
                (Y1,extra_genes,extra_Y) = self._read_gene_zs(gene_zs_in, **kwargs)
            elif exomes_in is not None:
                (Y1,extra_genes,extra_Y) = (np.zeros(Y1_exomes.shape), [], [])
            elif positive_controls_in is not None or positive_controls_list is not None:
                (Y1,extra_genes,extra_Y) = (np.zeros(Y1_positive_controls.shape), [], [])
            else:
                bail("Need to specify either gene_bfs_in or gene_percentiles_in or gene_zs_in or exomes_in or positive_controls_in")

            (Y1,extra_genes,extra_Y) = __hold_out_chrom(Y1,extra_genes,extra_Y)
            Y1_for_regression = copy.copy(Y1)
            extra_Y_for_regression = copy.copy(extra_Y)

        #we now need to construct several arrays
        #1. self.genes (if it hasn't been constructed already)
        #2. Y: the total combined gwas + exome values for all genes in self.genes
        #3. Y_exomes: the exome values for all genes in self.genes
        #4. extra_genes: genes not in self.genes but for which we have exome or gwas values
        #5. extra_Y: the total combined gwas + exome values for all genes in either gwas or exomes but not in self.genes
        #6. extra_Y_exomes: the total exome values for all genes in either gwas or exomes but not in self.genes

        if missing_value is None:
            if len(Y1) > 0:
                missing_value = np.nanmean(Y1)
            else:
                missing_value = 0

        if self.genes is None:
            assert(len(Y1) == 0)
            assert(len(Y1_exomes) == 0)
            assert(len(Y1_positive_controls) == 0)

            #combine everything
            genes_set = set(extra_genes).union(extra_genes_exomes_positive_controls)

            #really calling this just to set the genes
            self._set_X(self.X_orig, list(genes_set), self.gene_sets, skip_N=False)

            #now need to reorder
            Y = np.full(len(self.genes), missing_value, dtype=float)
            Y_for_regression = np.full(len(self.genes), missing_value, dtype=float)
            Y_exomes = np.full(len(self.genes), missing_value_exomes, dtype=float)
            Y_positive_controls = np.full(len(self.genes), missing_value_positive_controls, dtype=float)

            for i in range(len(extra_genes)):
                Y[self.gene_to_ind[extra_genes[i]]] = extra_Y[i]
                Y_for_regression[self.gene_to_ind[extra_genes[i]]] = extra_Y_for_regression[i]

            for i in range(len(extra_genes_exomes_positive_controls)):
                Y_exomes[self.gene_to_ind[extra_genes_exomes_positive_controls[i]]] = extra_Y_exomes[i]

            Y += Y_exomes
            Y += Y_positive_controls

            Y_for_regression += Y_exomes
            Y_for_regression += Y_positive_controls

            if self.huge_signal_bfs is not None or self.gene_covariates is not None:

                #we need to reorder
                #the order is the same as extra_genes
                if self.huge_signal_bfs is not None:
                    index_map = {i: self.gene_to_ind[extra_genes[i]] for i in range(len(extra_genes))}
                    self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, [index_map[x] for x in self.huge_signal_bfs.indices], self.huge_signal_bfs.indptr), shape=self.huge_signal_bfs.shape)

                if self.huge_signal_bfs_for_regression is not None:
                    index_map = {i: self.gene_to_ind[extra_genes[i]] for i in range(len(extra_genes))}
                    self.huge_signal_bfs_for_regression = sparse.csc_matrix((self.huge_signal_bfs_for_regression.data, [index_map[x] for x in self.huge_signal_bfs_for_regression.indices], self.huge_signal_bfs_for_regression.indptr), shape=self.huge_signal_bfs_for_regression.shape)

                if self.gene_covariates is not None:
                    index_map_rev = {self.gene_to_ind[extra_genes[i]]: i for i in range(len(extra_genes))}
                    if self.gene_covariates is not None:
                        self.gene_covariates = self.gene_covariates[[index_map_rev[x] for x in range(self.gene_covariates.shape[0])],:]

            extra_genes = []
            extra_Y = np.array([])
            extra_Y_for_regression = np.array([])
            extra_Y_exomes = np.array([])
            extra_Y_positive_controls = np.array([])

        else:
            #sum the existing genes together
            Y = Y1 + Y1_exomes + Y1_positive_controls
            Y[np.isnan(Y1)] = Y1_exomes[np.isnan(Y1)] + Y1_positive_controls[np.isnan(Y1)] + missing_value
            Y[np.isnan(Y1_exomes)] = Y1[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + missing_value_exomes
            Y[np.isnan(Y1_positive_controls)] = Y1[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + missing_value_positive_controls

            Y_for_regression = Y1_for_regression + Y1_exomes + Y1_positive_controls
            Y_for_regression[np.isnan(Y1_for_regression)] = Y1_exomes[np.isnan(Y1_for_regression)] + Y1_positive_controls[np.isnan(Y1_for_regression)] + missing_value
            Y_for_regression[np.isnan(Y1_exomes)] = Y1_for_regression[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + missing_value_exomes
            Y_for_regression[np.isnan(Y1_positive_controls)] = Y1_for_regression[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + missing_value_positive_controls

            Y_exomes = Y1_exomes
            Y_exomes[np.isnan(Y1_exomes)] = missing_value_exomes

            Y_positive_controls = Y1_positive_controls
            Y_positive_controls[np.isnan(Y1_positive_controls)] = missing_value_positive_controls

            extra_gene_to_ind = self._construct_map_to_ind(extra_genes)
            extra_Y = list(extra_Y)
            extra_Y_for_regression = list(extra_Y_for_regression)
            new_extra_Y_exomes = list(np.full(len(extra_Y), missing_value_exomes))
            new_extra_Y_positive_controls = list(np.full(len(extra_Y), missing_value_positive_controls))

            num_add = 0
            for i in range(len(extra_genes_exomes_positive_controls)):
                if extra_genes_exomes_positive_controls[i] in extra_gene_to_ind:
                    extra_Y[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    extra_Y_for_regression[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    new_extra_Y_exomes[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] = extra_Y_exomes[i]
                    new_extra_Y_positive_controls[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] = extra_Y_positive_controls[i]
                else:
                    num_add += 1
                    extra_genes.append(extra_genes_exomes_positive_controls[i])
                    extra_Y.append(extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    extra_Y_for_regression.append(extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    new_extra_Y_exomes.append(extra_Y_exomes[i])
                    new_extra_Y_positive_controls.append(extra_Y_positive_controls[i])

            extra_Y = np.array(extra_Y)
            extra_Y_for_regression = np.array(extra_Y_for_regression)
            extra_Y_exomes = np.array(new_extra_Y_exomes)
            extra_Y_positive_controls = np.array(new_extra_Y_positive_controls)

            if self.huge_signal_bfs is not None:
                #have to add space for the exomes results that were added at the end
                self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, self.huge_signal_bfs.indices, self.huge_signal_bfs.indptr), shape=(self.huge_signal_bfs.shape[0] + num_add, self.huge_signal_bfs.shape[1]))

            if self.huge_signal_bfs_for_regression is not None:
                #have to add space for the exomes results that were added at the end
                self.huge_signal_bfs_for_regression = sparse.csc_matrix((self.huge_signal_bfs_for_regression.data, self.huge_signal_bfs_for_regression.indices, self.huge_signal_bfs_for_regression.indptr), shape=(self.huge_signal_bfs_for_regression.shape[0] + num_add, self.huge_signal_bfs_for_regression.shape[1]))

            if self.gene_covariates is not None:
                add_gene_covariates = np.tile(np.mean(self.gene_covariates, axis=0), num_add).reshape((num_add, self.gene_covariates.shape[1]))
                self.gene_covariates = np.vstack((self.gene_covariates, add_gene_covariates))

        #Y contains all of the genes in self.genes that have gene statistics
        #extra_Y contains additional genes not in self.genes that have gene statistics.
        #Since these will be used in the regression, they must be accounted for in the normalization of X and V

        if len(extra_Y) > 0:
            Y = np.concatenate((Y, extra_Y))
            Y_for_regression = np.concatenate((Y_for_regression, extra_Y_for_regression))
            Y_exomes = np.concatenate((Y_exomes, extra_Y_exomes))
            Y_positive_controls = np.concatenate((Y_positive_controls, extra_Y_positive_controls))

        if self.X_orig is not None:
            #Use original X because no whitening has taken place yet
            log("Expanding matrix", TRACE)
            self._set_X(sparse.csc_matrix((self.X_orig.data, self.X_orig.indices, self.X_orig.indptr), shape=(self.X_orig.shape[0] + len(extra_Y), self.X_orig.shape[1])), self.genes, self.gene_sets, skip_V=True, skip_scale_factors=True, skip_N=False)

        if self.genes is not None:
            self._set_X(self.X_orig, self.genes + extra_genes, self.gene_sets, skip_N=False)


        self._set_Y(Y, Y_for_regression, Y_exomes, Y_positive_controls, skip_V=True, skip_scale_factors=True)

        #if we read in combined or priors
        if gene_combined_map is not None:
            self.combined_prior_Ys = copy.copy(self.Y)
            for i in range(len(self.genes)):
                if self.genes[i] in gene_combined_map:
                    self.combined_prior_Ys[i] = gene_combined_map[self.genes[i]]
        if gene_prior_map is not None:
            self.priors = np.zeros(len(self.genes))
            for i in range(len(self.genes)):
                if self.genes[i] in gene_prior_map:
                    self.priors[i] = gene_prior_map[self.genes[i]]


        #now for the covariates and correction

        if gene_covs_in is not None:
            #read in the covariates, ignoring any extra
            (cov_names, gene_covs, _, _) = self._read_gene_covs(gene_covs_in, **kwargs)
            cov_dirs = np.array([0]*len(cov_names))

            #mean imputation
            col_means = np.nanmean(gene_covs, axis=0)
            nan_indices = np.where(np.isnan(gene_covs))
            gene_covs[nan_indices] = np.take(col_means, nan_indices[1])

            #if it's already here, should have gene_covariates, gene_covariate_names, and gene_covariate_directions
            if self.gene_covariates is not None:
                assert(gene_covs.shape[0] == self.gene_covariates.shape[0])
                self.gene_covariates = np.hstack((self.gene_covariates, gene_covs))
                self.gene_covariate_names = self.gene_covariate_names + cov_names
                self.gene_covariate_directions = np.append(self.gene_covariate_directions, cov_dirs)
            else:
                self.gene_covariates = gene_covs
                self.gene_covariate_names = cov_names
                self.gene_covariate_directions = cov_dirs

        if self.gene_covariates is not None:

            #remove any constant features
            constant_features = np.isclose(np.var(self.gene_covariates, axis=0), 0)
            if np.sum(constant_features) > 0:
                self.gene_covariates = self.gene_covariates[:,~constant_features]
                self.gene_covariate_names = [self.gene_covariate_names[i] for i in np.where(~constant_features)[0]]
                self.gene_covariate_directions = np.array([self.gene_covariate_directions[i] for i in np.where(~constant_features)[0]])

            #remove correlated features
            prune_threshold = 0.95
            cor_mat = np.abs(np.corrcoef(self.gene_covariates.T))
            np.fill_diagonal(cor_mat, 0)

            while True:
                if np.max(cor_mat) < prune_threshold:
                    try:
                        np.linalg.inv(self.gene_covariates.T.dot(self.gene_covariates))
                        break
                    except np.linalg.LinAlgError:
                        pass

                #take the feature of the two toward the end
                max_index = np.unravel_index(np.argmax(cor_mat), cor_mat.shape)
                if np.max(max_index) == self.gene_covariate_intercept_index:
                    max_index = np.min(max_index)
                else:
                    max_index = np.max(max_index)
                log("Removing feature %s" % self.gene_covariate_names[max_index], TRACE)
                self.gene_covariates = np.delete(self.gene_covariates, max_index, axis=1)
                del self.gene_covariate_names[max_index]
                del self.gene_covariate_directions[max_index]

                cor_mat = np.delete(np.delete(cor_mat, max_index, axis=1), max_index, axis=0)
                if len(self.gene_covariates) == 0:
                    bail("Error: something went wrong with matrix inversion. Still couldn't invert after removing all but one column")

            #identify the intercept index
            self.gene_covariate_intercept_index = np.where(np.isclose(np.var(self.gene_covariates, axis=0), 0))[0]
            if len(self.gene_covariate_intercept_index) == 0:
                self.gene_covariates = np.hstack((self.gene_covariates, np.ones(self.gene_covariates.shape[0])[:,np.newaxis]))
                self.gene_covariate_names.append("intercept")
                self.gene_covariate_directions = np.append(self.gene_covariate_directions, 0)
                self.gene_covariate_intercept_index = len(self.gene_covariate_names) - 1
            else:
                self.gene_covariate_intercept_index = self.gene_covariate_intercept_index[0]

            #now add in the corrections
            #remove outliers on any of the metrics. We don't want to use them to fit the linear model
            covariate_means = np.mean(self.gene_covariates, axis=0)
            covariate_sds = np.std(self.gene_covariates, axis=0)
            #make sure intercept not excluded
            covariate_sds[covariate_sds == 0] = 1

            self.gene_covariates_mask = np.all(self.gene_covariates < covariate_means + 5 * covariate_sds, axis=1)
            self.gene_covariates_mat_inv = np.linalg.inv(self.gene_covariates[self.gene_covariates_mask,:].T.dot(self.gene_covariates[self.gene_covariates_mask,:]))
            gene_covariate_sds = np.std(self.gene_covariates, axis=0)
            gene_covariate_sds[gene_covariate_sds == 0] = 1
            self.gene_covariate_zs = (self.gene_covariates - np.mean(self.gene_covariates, axis=0)) / gene_covariate_sds

            #call correct betas

            Y_for_regression = self.Y_for_regression
            if self.Y_for_regression is not None:
                (Y_for_regression, _, _) = self._correct_huge(self.Y_for_regression, self.gene_covariates, self.gene_covariates_mask, self.gene_covariates_mat_inv, self.gene_covariate_names, self.gene_covariate_intercept_index)

            (Y, self.Y_uncorrected, _) = self._correct_huge(self.Y, self.gene_covariates, self.gene_covariates_mask, self.gene_covariates_mat_inv, self.gene_covariate_names, self.gene_covariate_intercept_index)

            self._set_Y(Y, Y_for_regression, self.Y_exomes, self.Y_positive_controls)

            self.gene_covariate_adjustments = self.Y_for_regression - self.Y_uncorrected

            #update original huge score data structures
            if self.gene_to_gwas_huge_score is not None:
                Y_huge = np.zeros(len(self.Y_for_regression))
                assert(len(Y_huge) == len(self.genes))
                for i in range(len(self.genes)):
                    if self.genes[i] in self.gene_to_gwas_huge_score:
                        Y_huge[i] = self.gene_to_gwas_huge_score[self.genes[i]]

                (Y_huge, Y_huge_uncorrected, _) = self._correct_huge(Y_huge, self.gene_covariates, self.gene_covariates_mask, self.gene_covariates_mat_inv, self.gene_covariate_names, self.gene_covariate_intercept_index)            

                for i in range(len(self.genes)):
                    if self.genes[i] in self.gene_to_gwas_huge_score:
                        self.gene_to_gwas_huge_score[self.genes[i]] = Y_huge[i]

                self.combine_huge_scores()


    #Initialize the matrices, genes, and gene sets
    #This can be called multiple times; it will subset the current matrices down to the new set of gene sets
    #any information regarding *genes* though is overwritten -- there is no way to subset the old genes down to a new set of genes
    #(although reading multiple files hasn't been tested thoroughly)
    def read_X(self, X_in, Xd_in=None, X_list=None, Xd_list=None, V_in=None, skip_V=True, force_reread=False, min_gene_set_size=1, max_gene_set_size=30000, only_ids=None, only_inc_genes=None, fraction_inc_genes=None, add_all_genes=False, prune_gene_sets=0.8, prune_deterministically=False, x_sparsify=[50,100,200,500,1000], add_ext=False, add_top=True, add_bottom=True, filter_negative=True, threshold_weights=0.5, cap_weights=True, permute_gene_sets=False, max_gene_set_p=None, filter_gene_set_p=1, increase_filter_gene_set_p=0.01, max_num_gene_sets_initial=None, max_num_gene_sets=None, skip_betas=False, run_logistic=True, max_for_linear=0.95, filter_gene_set_metric_z=2.5, initial_p=0.01, initial_sigma2=1e-3, initial_sigma2_cond=None, sigma_power=0, sigma_soft_threshold_95=None, sigma_soft_threshold_5=None, run_gls=False, run_corrected_ols=False, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, update_hyper_p=False, update_hyper_sigma=False, batch_all_for_hyper=False, first_for_hyper=False, first_max_p_for_hyper=False, first_for_sigma_cond=False, sigma_num_devs_to_top=2.0, p_noninf_inflate=1, batch_separator="@", ignore_genes=set(["NA"]), file_separator=None, max_num_burn_in=None, max_num_iter_betas=1100, min_num_iter_betas=10, num_chains_betas=10, r_threshold_burn_in_betas=1.01, use_max_r_for_convergence_betas=True, max_frac_sem_betas=0.01, max_allowed_batch_correlation=None, sparse_solution=False, sparse_frac_betas=None, betas_trace_out=None, show_progress=True):
        X_format = "<gene_set_id> <gene 1> <gene 2> ... <gene n>"
        V_format = "<gene_set1> <gene_set_2> ...<gene_set_n>\n<V11> <V12> ... <V1n>\n<V21> <V22> ... <V2n>"

        EXT_TAG = "ext"
        BOT_TAG = "bot"
        TOP_TAG = "top"

        if not force_reread and self.X_orig is not None:
            return

        self._set_X(None, self.genes, None, skip_N=True)

        self._record_params({"filter_gene_set_p": filter_gene_set_p, "filter_negative": filter_negative, "threshold_weights": threshold_weights, "cap_weights": cap_weights, "max_num_gene_sets_initial": max_num_gene_sets_initial, "max_num_gene_sets": max_num_gene_sets, "filter_gene_set_metric_z": filter_gene_set_metric_z, "num_chains_betas": num_chains_betas, "sigma_num_devs_to_top": sigma_num_devs_to_top, "p_noninf_inflate": p_noninf_inflate})

        def remove_tag(X_in, tag_separator=':'):
            tag = None
            if tag_separator in X_in:
                tag_index = X_in.index(tag_separator)
                tag = X_in[:tag_index]

                X_in = X_in[tag_index+1:]
                if len(tag) == 0:
                    tag = None
            return (X_in, tag)

        def expand_Xs(Xs, orig_files):
            new_Xs = []
            batches = []
            labels = []
            new_orig_files = []
            for i in range(len(Xs)):
                X = Xs[i]
                orig_file = orig_files[i]
                batch = None
                label = os.path.basename(orig_file)
                if "." in label:
                    label = ".".join(label.split(".")[:-1])
                if batch_separator in X:
                    batch = X.split(batch_separator)[-1]
                    label = batch
                    X = batch_separator.join(X.split(batch_separator)[:-1])

                (X, tag) = remove_tag(X)
                if tag is not None:
                    label = tag

                if file_separator is not None:
                    x_to_add = X.split(file_separator)
                else:
                    x_to_add = [X]

                new_Xs += x_to_add
                batches += [batch] * len(x_to_add)
                labels += [label] * len(x_to_add)
                new_orig_files += [orig_file] * len(x_to_add)
            return (new_Xs, batches, labels, new_orig_files)

        #list of the X files specified on the command line
        X_ins = []
        orig_files = []
        if X_in is not None:
            if type(X_in) == str:
                X_ins = [X_in]
                orig_files = [X_in]
            elif type(X_in) == list:
                X_ins = X_in
                orig_files = copy.copy(X_in)

        is_dense = []

        if X_list is not None:
            X_lists = []
            if type(X_list) == str:
                X_lists = [X_list]
            elif type(X_list) == list:
                X_lists = X_list

            for X_list in X_lists:
                batch = None
                if batch_separator in X_list:
                    batch = X_list.split(batch_separator)[-1]
                    X_list = batch_separator.join(X_list.split(batch_separator)[:-1])

                with open(X_list) as X_list_fh:
                    for line in X_list_fh:
                        line = line.strip()
                        if batch is not None and batch_separator not in line:
                            line = "%s%s%s" % (line, batch_separator, batch)
                        X_ins.append(line)
                        orig_files.append(X_list)

        X_ins, batches, labels, orig_files = expand_Xs(X_ins, orig_files)

        #TODO: read in labels here, batches2, and then when append

        is_dense = [False for x in X_ins]

        Xd_ins = []
        orig_dfiles = []
        if Xd_in is not None:
            if type(Xd_in) == str:
                Xd_ins = [Xd_in]
                orig_dfiles = [Xd_in]
            elif type(Xd_in) == list:
                Xd_ins = Xd_in
                orig_dfiles = Xd_in

        if Xd_list is not None:
            if type(Xd_list) == str:
                Xd_lists = [Xd_list]
            elif type(Xd_list) == list:
                Xd_lists = Xd_list

            for Xd_list in Xd_lists:
                batch = None
                if batch_separator in Xd_list:
                    batch = Xd_list.split(batch_separator)[-1]
                    Xd_list = batch_separator.join(Xd_list.split(batch_separator)[:-1])

                with open(Xd_list) as Xd_list_fh:
                    for line in Xd_list_fh:
                        line = line.strip()
                        if batch is not None and batch_separator not in line:
                            line = "%s%s%s" % (line, batch_separator, batch)
                        Xd_ins.append(line)
                        orig_dfiles.append(Xd_list)

        Xd_ins, batches2, labels2, orig_dfiles = expand_Xs(Xd_ins, orig_dfiles)

        X_ins += Xd_ins
        batches += batches2
        labels += labels2
        orig_files += orig_dfiles
        is_dense += [True for x in Xd_ins]

        #first reorder the files so that those with batches are at the front

        #X_ins = [X_ins[i] for i in range(len(batches)) if batches[i] is not None] + [X_ins[i] for i in range(len(batches)) if batches[i] is None]
        #is_dense = [is_dense[i] for i in range(len(batches)) if batches[i] is not None] + [is_dense[i] for i in range(len(batches)) if batches[i] is None]
        #orig_files = [orig_files[i] for i in range(len(batches)) if batches[i] is not None] + [orig_files[i] for i in range(len(batches)) if batches[i] is None]

        #batches = [batches[i] for i in range(len(batches)) if batches[i] is not None] + [batches[i] for i in range(len(batches)) if batches[i] is None]

        #README: batching / hyper semantics 
        #Use of @{batch} after file is the way to label a file
        #1. First we take each input file and assign it a batch
        #   If the file is labelled, that is the batch
        #   If the first file is not labelled, it is assigned a batch
        #   If the remaining files are not labelled, AND first-for-hyper is NOT set, they are assigned a batch.
        #   If first-for-hyper is set, batches with no labels bave None for batch
        #2. We then learn p (if update_hyper_p is specified) and sigma (if update_hyper_sigma is specified) separately for each batch
        #   All files with the same batch are pooled for learning the p and sigma
        #   If first_for_sigma_cond is specified, then the sigma to p ratio learned by the first batch is fixed throughout
        #   This means that if only one of sigma or p is learned, the other is adjusted to keep the sigma/p ratio the same

        #now handle the None batches
        #semantics are that things with a batch have value learned from all files with that batch,
        #things with None have it learned from first batch that appears in arg list

        used_batches = set([str(b) for b in batches if b is not None])
        next_batch_num = 1
        def __generate_new_batch(new_batch_num):
            new_batch = "BATCH%d" % new_batch_num
            while new_batch in used_batches:
                new_batch_num += 1
                new_batch = "BATCH%d" % new_batch_num
            used_batches.add(new_batch)
            return new_batch, new_batch_num

        for i in range(len(batches)):
            if batches[i] is None:
                batches[i], next_batch_num = __generate_new_batch(next_batch_num)

                if batch_all_for_hyper:
                    for j in range(i+1,len(batches)):
                        batches[j] = batches[i]
                    break
                else:
                    #now find all other none batches with the same file and update them too
                    for j in range(i+1,len(batches)):
                        if batches[j] is None and orig_files[i] == orig_files[j]:
                            batches[j] = batches[i]

            if first_for_hyper:
                #make sure though that at least one batch is not None (this is what we will use to learn everything)
                #but then break; keep None batches to learn from the first batch
                #also set all other batches to None (we won't be learning for those)
                for j in range(i+1, len(batches)):
                    if batches[j] != batches[i]:
                        batches[j] = None
                break


        self._record_params({"num_X_batches": len(batches)})

        if update_hyper_sigma or update_hyper_p:
            log("Will learn parameters for %d files as %d batches and fill in %d additional files from the first" % (len([x for x in batches if x is not None]), len(set([x for x in batches if x is not None])), len([x for x in batches if x is None])))
        if first_for_sigma_cond:
            log("Will fix conditional sigma from the first batch")
        #this will store the number of ignored gene sets per file
        num_ignored_gene_sets = np.zeros((len(batches)))

        #expands the file batches to have one per gene set
        self.gene_set_batches = np.array([])
        self.gene_set_labels = np.array([])

        self.gene_sets = []
        self.is_dense_gene_set = np.array([], dtype=bool)

        if (filter_gene_set_p < 1 or filter_gene_set_metric_z) and self.Y is not None:
            self.gene_sets_ignored = []
            if self.gene_set_labels is not None:
                self.gene_set_labels_ignored = np.array([])

            self.col_sums_ignored = np.array([])
            self.scale_factors_ignored = np.array([])
            self.mean_shifts_ignored = np.array([])
            self.beta_tildes_ignored = np.array([])
            self.p_values_ignored = np.array([])
            self.ses_ignored = np.array([])
            self.z_scores_ignored = np.array([])
            self.se_inflation_factors_ignored = np.array([])


            self.beta_tildes = np.array([])
            self.p_values = np.array([])
            self.ses = np.array([])
            self.z_scores = np.array([])

            self.se_inflation_factors = None

            self.total_qc_metrics = None
            self.mean_qc_metrics = None

            self.total_qc_metrics_missing = None
            self.mean_qc_metrics_missing = None

            self.total_qc_metrics_ignored = None
            self.mean_qc_metrics_ignored = None

            self.total_qc_metrics_directions = None


            self.ps = None
            self.ps_missing = None
            self.sigma2s = None
            self.sigma2s_missing = None


            if (run_gls or run_corrected_ols) and self.y_corr is None:
                correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)

                #convert X and Y to their new values
                min_correlation = 0.05
                self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, Y_corr_m=correlation_m, store_cholesky=run_gls, store_corr_sparse=run_corrected_ols, skip_V=True, skip_scale_factors=True, min_correlation=min_correlation)

        if not run_logistic and self.Y_for_regression is not None and np.max(np.exp(self.Y_for_regression + self.background_log_bf) / (1 + np.exp(self.Y_for_regression + self.background_log_bf))) > max_for_linear:
            log("Switching to logistic sampling due to high Y values", DEBUG)
            #run_logistic = True

        self._record_param("read_X_run_logistic", run_logistic)


        #returns num added, num ignored

        def __add_to_X(mat_info, genes, gene_sets, tag=None, skip_scale_factors=False, fname=None):

            #if self.genes_missing is not None:
            #    gene_to_ind = self._construct_map_to_ind(genes)
            #    #we are going to construct the full matrices including all of the missing genes
            #    #and then subset the matrix down
            #    genes += [x for x in self.genes_missing if x not in gene_to_ind]

            if tag is not None:
                gene_sets = ["%s_%s" % (tag, x) for x in gene_sets]

            is_dense = False
            if type(mat_info) is tuple:
                (data, row, col) = mat_info
                cur_X = sparse.csc_matrix((data, (row, col)), shape=(len(genes), len(gene_sets)))
                is_dense = False
                if cur_X.shape[1] == 0:
                    return (0, 0)

            else:

                #is_dense = True
                #disabling this setting
                is_dense = False

                if self.gene_label_map is not None:
                    genes = list(map(lambda x: self.gene_label_map[x] if x in self.gene_label_map else x, genes))

                #make sure no repeated genes
                if len(set(genes)) != len(genes):
                    #make the mask
                    seen_genes = set()
                    unique_mask = np.full(len(genes), True)

                    for i in range(len(genes)):
                        if genes[i] in seen_genes:
                            unique_mask[i] = False
                        else:
                            seen_genes.add(genes[i])
                    #now subset both down
                    mat_info = mat_info[unique_mask,:]
                    genes = [genes[i] for i in range(len(genes)) if unique_mask[i]] 


                #check if actually sparse
                if len(x_sparsify) > 0:
                    sparsity_threshold = 1 - np.max(x_sparsify).astype(float) / mat_info.shape[0]
                else:
                    sparsity_threshold = 0.95

                orig_dense_gene_sets = gene_sets

                cur_X = None
                #convert to sparse if (a) many zeros
                convert_to_sparse = np.sum(mat_info == 0, axis=0) / mat_info.shape[0] > sparsity_threshold

                # or (b) if all non-zero are same value
                abs_mat_info = np.abs(mat_info)
                max_weights = abs_mat_info.max(axis=0)
                all_non_zero_same = np.sum(abs_mat_info * (abs_mat_info != max_weights), axis=0) == 0

                convert_to_sparse = np.logical_or(convert_to_sparse, all_non_zero_same)
                if np.any(convert_to_sparse):
                    log("Detected sparse matrix for %d of %d columns" % (np.sum(convert_to_sparse), len(convert_to_sparse)), DEBUG)
                    cur_X = sparse.csc_matrix(mat_info[:,convert_to_sparse])
                    #update the gene sets, as well as the dense ones we will expand later
                    gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if convert_to_sparse[i]]
                    orig_dense_gene_sets = [orig_dense_gene_sets[i] for i in range(len(orig_dense_gene_sets)) if not convert_to_sparse[i]]

                    mat_info = mat_info[:,~convert_to_sparse]
                    #respect min gene size
                    enough_genes = self.get_col_sums(cur_X, num_nonzero=True) >= min_gene_set_size
                    if np.any(~enough_genes):
                        log("Excluded %d gene sets due to too small size" % np.sum(~enough_genes), DEBUG)
                        cur_X = cur_X[:,enough_genes]
                        gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if enough_genes[i]]

                if mat_info.shape[1] > 0:

                    mat_sd = np.std(mat_info, axis=0)
                    if np.any(mat_sd == 0):
                        mat_info = mat_info[:,mat_sd != 0]

                    mat_info = (mat_info - np.mean(mat_info, axis=0)) / np.std(mat_info, axis=0)

                    subset_mask = np.full(len(genes), True)
                    x_for_stats = mat_info
                    if self.Y is not None and self.genes is not None:
                        #make a mask to subset it down for the purposes of quantiles
                        subset_mask[[i for i in range(len(genes)) if genes[i] not in self.gene_to_ind]] = False
                        x_for_stats = mat_info[subset_mask,:]

                    if x_for_stats.shape[0] == 0:
                        warn("No genes in --Xd-in %swere seen before so skipping; example genes: %s" % ("%s " % fname if fname is not None else "", ",".join(genes[:4])))
                        return (0, 0)

                    top_numbers = list(reversed(sorted(x_sparsify)))

                    top_fractions = np.array(top_numbers, dtype=float) / x_for_stats.shape[0]

                    top_fractions[top_fractions > 1] = 1
                    top_fractions[top_fractions < 0] = 0

                    if len(top_fractions) == 0:
                        bail("No --X-sparsify set so doing nothing")
                        return (0, 0)

                    upper_quantiles = np.quantile(x_for_stats, 1 - top_fractions, axis=0)
                    lower_quantiles = np.quantile(x_for_stats, top_fractions, axis=0)

                    upper = copy.copy(mat_info)
                    lower = copy.copy(mat_info)

                    assert(np.all(upper_quantiles[0,:] == np.min(upper_quantiles, axis=0)))
                    assert(np.all(lower_quantiles[0,:] == np.max(lower_quantiles, axis=0)))

                    for i in range(len(top_numbers)):
                        #since we are sorted in descending order, can throw away everything below current threshold
                        upper_threshold_mask = upper < upper_quantiles[i,:]
                        if np.sum(upper_threshold_mask) == 0:
                            upper_threshold_mask = upper <= upper_quantiles[i,:]

                        lower_threshold_mask = lower > lower_quantiles[i,:]
                        if np.sum(lower_threshold_mask) == 0:
                            lower_threshold_mask = lower >= lower_quantiles[i,:]

                        mat_info[np.logical_and(upper_threshold_mask, lower_threshold_mask)] = 0
                        upper[upper_threshold_mask] = 0
                        lower[lower_threshold_mask] = 0

                        if add_ext:
                            temp_X = sparse.csc_matrix(mat_info)
                            top_gene_sets = ["%s_%s%d" % (x, EXT_TAG, top_numbers[i]) for x in orig_dense_gene_sets]
                            if cur_X is None:
                                cur_X = temp_X
                                gene_sets = top_gene_sets
                            else:
                                cur_X = sparse.hstack((cur_X, temp_X))
                                gene_sets = gene_sets + top_gene_sets

                        if add_bottom:
                            temp_X = sparse.csc_matrix(lower)
                            top_gene_sets = ["%s_%s%d" % (x, BOT_TAG, top_numbers[i]) for x in orig_dense_gene_sets]
                            if cur_X is None:
                                cur_X = temp_X
                                gene_sets = top_gene_sets
                            else:
                                cur_X = sparse.hstack((cur_X, temp_X))
                                gene_sets = gene_sets + top_gene_sets

                        if add_top or (not add_ext and not add_bottom):
                            temp_X = sparse.csc_matrix(upper)
                            top_gene_sets = ["%s_%s%d" % (x, TOP_TAG, top_numbers[i]) for x in orig_dense_gene_sets]
                            if cur_X is None:
                                cur_X = temp_X
                                gene_sets = top_gene_sets
                            else:
                                gene_sets = gene_sets + top_gene_sets
                                cur_X = sparse.hstack((cur_X, temp_X))

                        if cur_X is None:
                            return (0, 0)

                        #if all of the values for a row are negative, flip the sign to make it positive
                        all_negative_mask = ((cur_X < 0).sum(axis=0) == cur_X.astype(bool).sum(axis=0)).A1
                        cur_X[:,all_negative_mask] = -cur_X[:,all_negative_mask]

                        cur_X.eliminate_zeros()

                    if cur_X is None or cur_X.shape[1] == 0:
                        return (0, 0)

                if self.genes is not None:
                    #need to reorder the genes to match the old and add the new ones
                    old_genes = genes
                    genes = self.genes
                    if self.genes_missing is not None:
                        genes += self.genes_missing
                    genes += [x for x in old_genes if (self.gene_to_ind is None or x not in self.gene_to_ind) and (self.gene_missing_to_ind is None or x not in self.gene_missing_to_ind)]
                    gene_to_ind = self._construct_map_to_ind(genes)
                    index_map = {i: gene_to_ind[old_genes[i]] for i in range(len(old_genes))}
                    cur_X = sparse.csc_matrix((cur_X.data, [index_map[x] for x in cur_X.indices], cur_X.indptr), shape=(len(genes), cur_X.shape[1]))

            denom = self.get_col_sums(cur_X, num_nonzero=True)
            denom[denom == 0] = 1
            avg_weights = np.abs(cur_X).sum(axis=0) / denom
            if np.sum(avg_weights != 1) > 0:
                #(mean_shifts_raw, scale_factors_raw) = self._calc_X_shift_scale(cur_X)
                #(mean_shifts_bool, scale_factors_bool) = self._calc_X_shift_scale(cur_X.astype(bool))
                #avg_weights = scale_factors_raw / scale_factors_bool

                #this is an option to use the max weight after throwing out outliers as the norm
                #it doesn't look to work as well as avg_weights
                max_weight_devs = None
                if max_weight_devs is not None:
                    dev_weights = np.sqrt(np.abs(cur_X).power(2).sum(axis=0) / denom - np.power(avg_weights, 2))
                    temp_X = copy.copy(np.abs(cur_X))
                    temp_X[temp_X > avg_weights + max_weight_devs * dev_weights] = 0

                    #I don't think we need to set really low ones to zero, since temp_X is only positive
                    #temp_X[temp_X < avg_weights - max_weight_devs * dev_weights] = 0

                    weight_norm = temp_X.max(axis=0).todense().A1
                else:
                    weight_norm = avg_weights.A1

                weight_norm = np.round(weight_norm, 10)
                weight_norm[weight_norm == 0] = 1

                #assume rows are already normalized if (a) all are below 1 and (b) threshold is None or all are above threshold 
                #so, normalize if (a) any is above 1 or (b) threshold is not None and any are below threshold 
                normalize_mask = (np.abs(cur_X) > 1).sum(axis=0).A1 > 0
                if threshold_weights is not None and threshold_weights > 0:
                    #check for those that have different number above 0 and above threshold
                    normalize_mask = np.logical_or(normalize_mask, (np.abs(cur_X) >= threshold_weights).sum(axis=0).A1 != (np.abs(cur_X) > 0).sum(axis=0).A1)

                #this uses less memory
                weight_norm[~normalize_mask] = 1.0
                cur_X = sparse.csc_matrix(cur_X.multiply(1.0 / weight_norm))
                #old method that uses higher memory
                #cur_X[:,normalize_mask] = sparse.csc_matrix(cur_X[:,normalize_mask].multiply(1.0 / weight_norm[normalize_mask]))

                #don't do binary; use threshold instead
                #if make_binary_weights is not None:
                #    cur_X.data[np.abs(cur_X.data) < make_binary_weights] = 0
                #    cur_X.data[np.abs(cur_X.data) >= make_binary_weights] = 1

                if threshold_weights is not None and threshold_weights > 0:
                    cur_X.data[np.abs(cur_X.data) < threshold_weights] = 0
                    if cap_weights:
                        cur_X.data[cur_X.data > 1] = 1
                        cur_X.data[cur_X.data < -1] = -1
                cur_X.eliminate_zeros()
            #now need to find any new genes that will be added as missing later, as well as any missing genes that need to be updated

            gene_ignored_N = None

            #these are the new missing that are in the old missing
            #these are not necessarily in the self.X structures, since self.genes could be set before that
            genes_missing_int = []
            cur_X_missing_genes_int = None
            gene_ignored_N_missing_int = None

            #these are the new missing that are not in the old missing
            genes_missing_new = []
            cur_X_missing_genes_new = None
            gene_ignored_N_missing_new = None


            if (self.Y is not None and len(genes) > len(self.Y)) or (only_inc_genes is not None and self.genes is not None):
                genes_missing_old = self.genes_missing if self.genes_missing is not None else []
                gene_missing_old_to_ind = self._construct_map_to_ind(genes_missing_old)
                gene_to_ind = self._construct_map_to_ind(genes)

                #these are the genes that are new this time around
                genes_missing_new = [x for x in genes if x not in self.gene_to_ind and x not in gene_missing_old_to_ind]
                genes_missing_new_set = set(genes_missing_new)

                #these are missing genes shared with before
                genes_missing_int = [x for x in genes if x in gene_missing_old_to_ind]
                genes_missing_int_set = set(genes_missing_int)

                #all genes missing
                #genes_missing = set(genes_missing_new + genes_missing_int + genes_missing_old)
                #gene_missing_to_ind = self._construct_map_to_ind(genes_missing)
                #assert(len(genes_missing) == len(set(genes_missing)))

                #subset down X to only non missing

                int_mask = np.full(len(genes), False)
                int_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_int_set]] = True
                if np.sum(int_mask) > 0:
                    cur_X_missing_genes_int = cur_X[int_mask,:]

                new_mask = np.full(len(genes), False)
                new_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_new_set]] = True
                if np.sum(new_mask) > 0:
                    cur_X_missing_genes_new = cur_X[new_mask,:]

                subset_mask = np.full(len(genes), True)
                subset_mask[[i for i in range(len(genes)) if genes[i] not in self.gene_to_ind]] = False

                cur_X = cur_X[subset_mask,:]

                genes = [x for x in genes if x in self.gene_to_ind]

                #remove empty gene sets
                gene_set_nonempty_mask = self.get_col_sums(cur_X) > 0

                if np.sum(~gene_set_nonempty_mask) > 0:
                    cur_X = cur_X[:,gene_set_nonempty_mask]

                    if cur_X_missing_genes_int is not None:
                        cur_X_missing_genes_int = cur_X_missing_genes_int[:,gene_set_nonempty_mask]
                    if cur_X_missing_genes_new is not None:
                        cur_X_missing_genes_new = cur_X_missing_genes_new[:,gene_set_nonempty_mask]

                    gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if gene_set_nonempty_mask[i]]

                #at this point, we have subset X down to only non missing genes before
                if self.Y is not None:
                    assert(len(genes) == len(self.Y))

                if cur_X.shape[1] == 0:
                    bail("Error: no gene sets overlapped Y and X; you may have forgotten to map gene names over to a common namespace")

                #our missing genes come from two sources: self.X_orig_missing_genes (those are the old ones) and cur_X_missing_gnenes (those are the new ones). genes_missing_to_add tells us which are new

                #we only added genes at the end
                #num_add = len(genes) - len(self.Y)

                #new_Y = np.append(self.Y, np.full(num_add, np.nanmean(self.Y)))
                #new_Y_exomes = self.Y_exomes
                #if self.Y_exomes is not None:
                #    new_Y_exomes = np.append(self.Y_exomes, np.full(num_add, np.nanmean(self.Y_exomes)))

                #if self.y_corr is not None:
                #    padding = np.zeros((self.y_corr.shape[0], num_add))
                #    padding[0,:] = 1
                #    self.y_corr = np.hstack((self.y_corr, padding))

                #self._set_Y(new_Y, new_Y_exomes, Y_corr_m=self.y_corr, store_cholesky=run_gls and num_add > 0, store_corr_sparse=run_corrected_ols and num_add > 0, skip_V=skip_V)

                #if self.huge_signal_bfs is not None:
                #    self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, self.huge_signal_bfs.indices, self.huge_signal_bfs.indptr), shape=(self.huge_signal_bfs.shape[0] + num_add, self.huge_signal_bfs.shape[1]))


            if permute_gene_sets:
                #build random permutation
                #have to do this at the end after we have added in all of the genes (even those missing from the original X)
                if self.Y is not None:
                    assert(len(self.Y) == len(self.genes))
                    #self.genes is always at the start
                    #permute only within those that have non-missing data
                    orig_indices = list(range(len(self.Y)))
                    new_indices = random.sample(orig_indices, len(orig_indices))
                    if cur_X.shape[0] > len(orig_indices):
                        num_to_add = cur_X.shape[0] - len(orig_indices)
                        to_add = list(range(len(orig_indices), len(orig_indices) + num_to_add))
                        orig_indices += to_add
                        new_indices += random.sample(to_add, len(to_add))
                else:
                    orig_indices = list(range(cur_X.shape[0]))
                    new_indices = random.sample(orig_indices, len(orig_indices))

                index_map = dict(zip(orig_indices, new_indices))
                cur_X = sparse.csc_matrix(cur_X)
                cur_X = sparse.csc_matrix((cur_X.data, [index_map[x] for x in cur_X.indices], cur_X.indptr), shape=(cur_X.shape[0], cur_X.shape[1]))                

            p_value_ignore = None

            if (filter_gene_set_p < 1 or filter_gene_set_metric_z is not None) and self.Y is not None:

                log("Analyzing gene sets to pre-filter")

                (mean_shifts, scale_factors) = self._calc_X_shift_scale(cur_X)

                total_qc_metrics = None
                mean_qc_metrics = None                
                total_qc_metrics_directions = None
                if self.gene_covariates is not None:
                    cur_X_size = np.abs(cur_X).sum(axis=0)
                    cur_X_size[cur_X_size == 0] = 1

                    total_qc_metrics = (np.array(cur_X.T.dot(self.gene_covariate_zs).T / cur_X_size)).T
                    total_qc_metrics = np.hstack((total_qc_metrics[:,:self.gene_covariate_intercept_index], total_qc_metrics[:,self.gene_covariate_intercept_index+1:]))

                    total_qc_metrics_directions = np.append(self.gene_covariate_directions[:self.gene_covariate_intercept_index], self.gene_covariate_directions[self.gene_covariate_intercept_index+1:])

                    total_huge_adjustments = (np.array(cur_X.T.dot(self.gene_covariate_adjustments).T / cur_X_size)).T

                    total_qc_metrics = np.hstack((total_qc_metrics, total_huge_adjustments))
                    total_qc_metrics_directions = np.append(total_qc_metrics_directions, -1)

                    if options.debug_only_avg_huge:
                        total_qc_metrics = total_huge_adjustments
                        total_qc_metrics_directions = np.array(-1)

                    mean_qc_metrics = total_huge_adjustments.squeeze()
                    mean_qc_metrics = total_huge_adjustments
                    if len(mean_qc_metrics.shape) == 2 and mean_qc_metrics.shape[1] == 1:
                        mean_qc_metrics = mean_qc_metrics.squeeze(axis=1)

                Y_to_use = self.Y_for_regression

                if run_logistic:
                    Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(cur_X, Y, scale_factors, mean_shifts, resid_correlation_matrix=self.y_corr_sparse)
                else:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._compute_beta_tildes(cur_X, Y_to_use, np.var(Y_to_use), scale_factors, mean_shifts, resid_correlation_matrix=self.y_corr_sparse)


                #if we have negative weights, that means we don't know which side is actually "better" for the trait (the feature is continuous). So flip the sign if the beta is negative
                negative_weights_mask = (cur_X < 0).sum(axis=0).A1 > 0
                if np.sum(negative_weights_mask) > 0:
                    flip_mask = np.logical_and(beta_tildes < 0, negative_weights_mask)
                    if np.sum(flip_mask) > 0:
                        log("Flipped %d gene sets" % np.sum(flip_mask), DEBUG)
                        beta_tildes[flip_mask] = -beta_tildes[flip_mask]
                        z_scores[flip_mask] = -z_scores[flip_mask]
                        cur_X[:,flip_mask] = -cur_X[:,flip_mask]

                p_value_mask = p_values <= filter_gene_set_p

                if increase_filter_gene_set_p is not None and np.mean(p_value_mask) < increase_filter_gene_set_p:
                    #choose a new more lenient threshold
                    p_from_quantile = np.quantile(p_values, increase_filter_gene_set_p)
                    log("Choosing revised p threshold %.3g to ensure keeping %.3g fraction of gene sets" % (p_from_quantile, increase_filter_gene_set_p), DEBUG)
                    p_value_mask = p_values <= p_from_quantile

                    if np.sum(~p_value_mask) > 0:
                        log("Ignoring %d gene sets due to p-value filters" % (np.sum(~p_value_mask)))

                if filter_negative:
                    negative_beta_tildes_mask = beta_tildes < 0
                    p_value_mask = np.logical_and(p_value_mask, ~negative_beta_tildes_mask)
                    if np.sum(negative_beta_tildes_mask) > 0:
                        log("Ignoring %d gene sets due to negative beta filters" % (np.sum(negative_beta_tildes_mask)))

                p_value_ignore = np.full(len(p_value_mask), False)
                if filter_gene_set_p < 1 or filter_gene_set_metric_z is not None:

                    p_value_ignore = ~p_value_mask
                    if np.sum(p_value_ignore) > 0:
                        log("Kept %d gene sets after p-value and beta filters" % (np.sum(p_value_mask)))

                    self.gene_sets_ignored = self.gene_sets_ignored + [gene_sets[i] for i in range(len(gene_sets)) if p_value_ignore[i]]
                    gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if p_value_mask[i]]

                    self.col_sums_ignored = np.append(self.col_sums_ignored, self.get_col_sums(cur_X[:,p_value_ignore]))
                    self.scale_factors_ignored = np.append(self.scale_factors_ignored, scale_factors[p_value_ignore])
                    self.mean_shifts_ignored = np.append(self.mean_shifts_ignored, mean_shifts[p_value_ignore])
                    self.beta_tildes_ignored = np.append(self.beta_tildes_ignored, beta_tildes[p_value_ignore])
                    self.p_values_ignored = np.append(self.p_values_ignored, p_values[p_value_ignore])
                    self.ses_ignored = np.append(self.ses_ignored, ses[p_value_ignore])
                    self.z_scores_ignored = np.append(self.z_scores_ignored, z_scores[p_value_ignore])

                    self.beta_tildes = np.append(self.beta_tildes, beta_tildes[p_value_mask])
                    self.p_values = np.append(self.p_values, p_values[p_value_mask])
                    self.ses = np.append(self.ses, ses[p_value_mask])
                    self.z_scores = np.append(self.z_scores, z_scores[p_value_mask])

                    if se_inflation_factors is not None:
                        self.se_inflation_factors_ignored = np.append(self.se_inflation_factors_ignored, se_inflation_factors[p_value_ignore])
                        if self.se_inflation_factors is None:
                            self.se_inflation_factors = np.array([])
                        self.se_inflation_factors = np.append(self.se_inflation_factors, se_inflation_factors[p_value_mask])

                    if self.gene_covariates is not None:
                        if self.total_qc_metrics_ignored is None:
                            self.total_qc_metrics_ignored = total_qc_metrics[p_value_ignore,:]
                            self.mean_qc_metrics_ignored = mean_qc_metrics[p_value_ignore]
                        else:
                            self.total_qc_metrics_ignored = np.vstack((self.total_qc_metrics_ignored, total_qc_metrics[p_value_ignore,:]))
                            self.mean_qc_metrics_ignored = np.append(self.mean_qc_metrics_ignored, mean_qc_metrics[p_value_ignore])

                        total_qc_metrics = total_qc_metrics[p_value_mask]
                        mean_qc_metrics = mean_qc_metrics[p_value_mask]                        

                    #need to record how many ignored
                    gene_ignored_N = self.get_col_sums(cur_X[:,p_value_ignore], axis=1)

                    if cur_X_missing_genes_new is not None:
                        gene_ignored_N_missing_new = np.array(np.abs(cur_X_missing_genes_new[:,p_value_ignore]).sum(axis=1)).flatten()
                        cur_X_missing_genes_new = cur_X_missing_genes_new[:,p_value_mask]

                    if cur_X_missing_genes_int is not None:
                        gene_ignored_N_missing_int = np.array(np.abs(cur_X_missing_genes_int[:,p_value_ignore]).sum(axis=1)).flatten()
                        cur_X_missing_genes_int = cur_X_missing_genes_int[:,p_value_mask]

                    cur_X = cur_X[:,p_value_mask]

            #construct the mean shifts / etc needed for compute beta tildes
            #then call compute beta tildes
            #then call compute betas without V
            #then filter

            self.is_dense_gene_set = np.append(self.is_dense_gene_set, np.full(len(gene_sets), is_dense))

            num_new_gene_sets = len(gene_sets)
            num_old_gene_sets = len(self.gene_sets) if self.gene_sets is not None else 0
            if self.X_orig is not None:

                cur_X = sparse.hstack((self.X_orig, cur_X))
                gene_sets = self.gene_sets + gene_sets

            if self.genes_missing is not None:
                genes += self.genes_missing

                if self.X_orig_missing_genes is None:
                    X_orig_missing_genes = sparse.csc_matrix(([], ([], [])), shape=(len(self.genes_missing), num_old_gene_sets))
                else:
                    X_orig_missing_genes = copy.copy(self.X_orig_missing_genes)                    

                if cur_X_missing_genes_int is not None:
                    if self.gene_ignored_N_missing is not None:
                        if gene_ignored_N_missing_int is not None:
                            self.gene_ignored_N_missing += gene_ignored_N_missing_int
                    else:
                        self.gene_ignored_N_missing = gene_ignored_N_missing_int

                    cur_X = sparse.vstack((cur_X, sparse.hstack((X_orig_missing_genes, cur_X_missing_genes_int))))
                elif X_orig_missing_genes is not None:
                    X_orig_missing_genes.resize((X_orig_missing_genes.shape[0], X_orig_missing_genes.shape[1] + num_new_gene_sets))
                    cur_X = sparse.vstack((cur_X, X_orig_missing_genes))

            if cur_X_missing_genes_new is not None:
                cur_X = sparse.vstack((cur_X, sparse.hstack((sparse.csc_matrix(([], ([], [])), shape=(cur_X_missing_genes_new.shape[0], num_old_gene_sets)), cur_X_missing_genes_new))))
                if self.gene_ignored_N_missing is not None:
                    if gene_ignored_N_missing_new is not None:
                        self.gene_ignored_N_missing = np.append(self.gene_ignored_N_missing, gene_ignored_N_missing_new)
                else:
                    self.gene_ignored_N_missing = gene_ignored_N_missing_new

                genes += genes_missing_new

            #save subset mask for later
            subset_mask = np.full(len(genes), True)
            if self.gene_to_ind is not None:

                subset_mask[[i for i in range(len(genes)) if genes[i] not in self.gene_to_ind]] = False

            #set full X with including new and old missing genes

            num_added = cur_X.shape[1]
            if self.X_orig is not None:
                num_added -= self.X_orig.shape[1]
            num_ignored = np.sum(p_value_ignore) if p_value_ignore is not None else 0

            self._set_X(sparse.csc_matrix(cur_X, shape=cur_X.shape), genes, gene_sets, skip_scale_factors=skip_scale_factors, skip_V=True, skip_N=False)

            #have to add ignored_N since this is only place we have the information
            if self.gene_ignored_N is not None:
                if gene_ignored_N is not None:
                    self.gene_ignored_N += gene_ignored_N
            else:
                self.gene_ignored_N = gene_ignored_N

            if self.gene_ignored_N is not None and self.gene_ignored_N_missing is not None:
                self.gene_ignored_N = np.append(self.gene_ignored_N, self.gene_ignored_N_missing)

            #have to call this function to ensure every data structure gets subsetted
            #don't subset Y since we didn't expand these

            self._subset_genes(subset_mask, skip_V=True, overwrite_missing=True, skip_scale_factors=False, skip_Y=True)

            if self.gene_covariates is not None:
                if self.total_qc_metrics is None:
                    self.total_qc_metrics = total_qc_metrics
                    self.mean_qc_metrics = mean_qc_metrics
                else:
                    self.total_qc_metrics = np.vstack((self.total_qc_metrics, total_qc_metrics))
                    self.mean_qc_metrics = np.append(self.mean_qc_metrics, mean_qc_metrics)

                self.total_qc_metrics_directions = total_qc_metrics_directions

            return (num_added, num_ignored)

        ignored_gs = 0

        if only_inc_genes:
            add_all_genes = True

        if self.genes is None or add_all_genes:
            if self.genes is None:
                log("No genes initialized before reading X: constructing gene list from union of all files", DEBUG)
            #need to set it to the union of all genes
            all_genes = []
            gene_counts = {}
            num_gene_sets = 0
            for i in range(len(X_ins)):
                X_in = X_ins[i]
                (X_in, tag) = remove_tag(X_in)

                if is_dense[i]:
                    with open_gz(X_in) as gene_sets_fh:
                        num_in_file = None
                        for line in gene_sets_fh:
                            line = line.strip()
                            cols = line.split()
                            if num_in_file is None:
                                num_in_file = len(cols) - 1
                                num_gene_sets += num_in_file
                            elif len(cols) - 1 != num_in_file:
                                bail("Not a square matrix!")

                            if len(cols) > 0:
                                all_genes += cols[0]
                            if cols[0] not in gene_counts:
                                    gene_counts[cols[0]] = 0
                            gene_counts[cols[0]] += num_in_file
                else:
                    with open_gz(X_in) as gene_sets_fh:
                        it = 0
                        for line in gene_sets_fh:
                            line = line.strip()
                            cols = line.split()
                            if len(cols) < 2:
                                continue

                            cur_genes = set(cols[1:])

                            if only_ids is not None and cols[0] not in only_ids:
                                continue

                            if ":" in line:
                                cur_genes = [gene.split(":")[0] for gene in cur_genes]
                            if self.gene_label_map is not None:
                                cur_genes = set(map(lambda x: self.gene_label_map[x] if x in self.gene_label_map else x, cur_genes))

                            if not add_all_genes and only_inc_genes is not None:
                                fraction_match = len(only_inc_genes.intersection(cur_genes)) / float(len(only_inc_genes))

                                if fraction_match < (fraction_inc_genes if fraction_inc_genes is not None else 1e-5):
                                    continue

                            all_genes += cur_genes
                            for gene in cur_genes:
                                if gene not in gene_counts:
                                    gene_counts[gene] = 0
                                gene_counts[gene] += 1

                            num_gene_sets += 1
                            it += 1
                            if it % 1000 == 0:
                                all_genes = list(set(all_genes))

                all_genes = list(set(all_genes))

            if self.genes is not None:
                add_genes = [x for x in all_genes if x not in self.gene_to_ind]
                log("Adding an additional %d genes from gene sets not in input Y values" % len(add_genes), DEBUG)
                all_genes = self.genes + add_genes
                new_Y = self.Y
                if new_Y is not None:
                    assert(len(new_Y) == len(self.genes))
                    new_Y = np.append(new_Y, np.zeros(len(add_genes)))
                new_Y_for_regression = self.Y_for_regression
                if new_Y_for_regression is not None:
                    assert(len(new_Y_for_regression) == len(self.genes))
                    new_Y_for_regression = np.append(new_Y_for_regression, np.zeros(len(add_genes)))
                new_Y_exomes = self.Y_exomes
                if new_Y_exomes is not None:
                    assert(len(new_Y_exomes) == len(self.genes))
                    new_Y_exomes = np.append(new_Y_exomes, np.zeros(len(add_genes)))
                new_Y_positive_controls = self.Y_positive_controls
                if new_Y_positive_controls is not None:
                    assert(len(new_Y_positive_controls) == len(self.genes))
                    new_Y_positive_controls = np.append(new_Y_positive_controls, np.zeros(len(add_genes)))
                self._set_Y(new_Y, new_Y_for_regression, new_Y_exomes, new_Y_positive_controls)

            #really calling this just to set the genes
            self._set_X(self.X_orig, list(all_genes), self.gene_sets, skip_N=False)

        for i in range(len(X_ins)):
            X_in = X_ins[i]
            (X_in, tag) = remove_tag(X_in)

            log("Reading X %d of %d from --X-in file %s" % (i+1,len(X_ins),X_in), INFO)

            num_too_small = 0

            genes = []
            gene_sets = []
            cur_X = None

            ignored_for_fraction_inc = 0

            if is_dense[i]:
                with open_gz(X_in) as gene_sets_fh:
                    header = gene_sets_fh.readline().strip()
                    header = header.lstrip("# \t")
                    gene_sets = header.split()
                    if len(gene_sets) < 2:
                        warn("First line of --Xd-in %s must contain gene column followed by list of gene sets; skipping file" % X_in)
                        continue
                    #if header[0] != "#":
                    #    warn("Assuming first line is header line despite lack of #; first characters are '%s...'" % header[:10])

                    #first column is genes so split
                    gene_sets = gene_sets[1:]

                    #maximum number of sets to avoid memory overflow
                    max_num_at_once = 500
                    if only_ids and len(only_ids) < len(gene_sets):
                        #estimate fraction
                        max_num_at_once = int(max_num_at_once / (float(len(only_ids)) / len(gene_sets)))

                    if len(gene_sets) > max_num_at_once:
                        log("Splitting reading of file into chunks to limit memory", DEBUG)
                    for j in range(0, len(gene_sets), max_num_at_once):
                        if len(gene_sets) > max_num_at_once:
                            log("Reading gene sets %d-%d" % (j+1, j+min(len(gene_sets), j+max_num_at_once+1)), DEBUG)

                        gene_set_indices_to_load = list(range(j, min(len(gene_sets), j+max_num_at_once)))

                        if only_ids is not None:
                            gene_set_mask = np.full(len(gene_set_indices_to_load), False)
                            for k in range(len(gene_set_mask)):
                                if gene_sets[gene_set_indices_to_load[k]] in only_ids:
                                    gene_set_mask[k] = True
                                elif x_sparsify is not None:
                                    for top_number in x_sparsify:
                                        for sparse_tag in [EXT_TAG, TOP_TAG, BOT_TAG]:
                                            if "%s_%s%d" % (gene_sets[gene_set_indices_to_load[k]], sparse_tag, top_number) in only_ids:
                                                gene_set_mask[k] = True
                                                break

                            if np.any(gene_set_mask):
                                gene_set_indices_to_load = [gene_set_indices_to_load[i] for i in range(len(gene_set_mask)) if gene_set_mask[i]]
                                log("Will load %d gene sets that were requested" % (np.sum(gene_set_mask)), TRACE)
                            else:
                                continue

                        indices_to_load = [0] + [k+1 for k in gene_set_indices_to_load]

                        cur_X = np.loadtxt(X_in, skiprows=1, dtype=str, usecols=indices_to_load)

                        if len(cur_X.shape) == 1:
                            cur_X = cur_X[:,np.newaxis]

                        if cur_X.shape[1] != len(indices_to_load):
                            bail("Xd matrix %s dimensions %s do not match number of gene sets in header line (%s)" % (X_in, cur_X.shape, len(gene_sets)))
                        cur_gene_sets = [gene_sets[k] for k in gene_set_indices_to_load]

                        genes = cur_X[:,0]
                        if self.gene_label_map is not None:
                            genes = list(map(lambda x: self.gene_label_map[x] if x in self.gene_label_map else x, genes))

                        cur_X = cur_X[:,1:].astype(float)
                        mat_info = cur_X

                        num_added, num_ignored = __add_to_X(mat_info, genes, cur_gene_sets, tag, skip_scale_factors=False)
                        if i == 0 and num_added + num_ignored == 0:
                            bail("--first-for-hyper was specified but first file had no gene sets")
                        #add gene set batches here
                        self.gene_set_batches = np.append(self.gene_set_batches, np.full(num_added, batches[i]))
                        self.gene_set_labels = np.append(self.gene_set_labels, np.full(num_added, labels[i]))
                        self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, np.full(num_ignored, labels[i]))
                        num_ignored_gene_sets[i] += num_ignored

            else:

                data = []
                row = []
                col = []
                num_read = 0

                new_gene_to_ind = {}
                gene_set_to_ind = {}
                gene_to_ind = None
                if self.genes is not None:
                    #ensure that the matrix always contains all of the current genes
                    #this simplifies code in add_to_X
                    genes = copy.copy(self.genes)
                    if self.genes_missing is not None:
                        genes += self.genes_missing
                    gene_to_ind = self._construct_map_to_ind(genes)

                with open_gz(X_in) as gene_sets_fh:

                    max_num_entries_at_once = 200 * 10000
                    cur_num_read = 0

                    already_seen = 0
                    for line in gene_sets_fh:
                        line = line.strip()
                        cols = line.split()

                        if len(cols) < 2:
                            warn("Line does not match format for --X-in: %s" % (line))
                            continue
                        gs = cols[0]

                        if only_ids is not None and gs not in only_ids:
                            continue

                        if gs in gene_set_to_ind or (self.gene_set_to_ind is not None and gs in self.gene_set_to_ind):
                            already_seen += 1
                            continue

                        cur_genes = set(cols[1:])
                        if self.gene_label_map is not None:
                            cur_genes = set(map(lambda x: self.gene_label_map[x] if x in self.gene_label_map else x, cur_genes))

                        if len(cur_genes) < min_gene_set_size:
                            #avoid too small gene sets
                            num_too_small += 1
                            continue

                        #initialize a new location for the gene set
                        gene_set_ind = len(gene_sets)
                        gene_sets.append(gs)
                        #add this to track duplicates in input file
                        gene_set_to_ind[gs] = gene_set_ind

                        if only_inc_genes is not None:
                            fraction_match = len(only_inc_genes.intersection(cur_genes)) / float(len(only_inc_genes))
                            if fraction_match < (fraction_inc_genes if fraction_inc_genes is not None else 1e-5):
                                ignored_for_fraction_inc += 1
                                continue

                        for gene in cur_genes:

                            gene_array = gene.split(":")
                            gene = gene_array[0]
                            if gene in ignore_genes:
                                continue
                            if len(gene_array) == 2:
                                try:
                                    weight = float(gene_array[1])
                                except ValueError:
                                    #skip this line
                                    warn("Couldn't convert weight %s to number so skipping token: %s" % (weight, ":".join(gene_array)))
                                    continue
                            else:
                                weight = 1.0

                            if gene_to_ind is not None and gene in gene_to_ind:
                                #keep this gene when we harmonize at the end
                                gene_ind = gene_to_ind[gene]
                            else:
                                if gene not in new_gene_to_ind:
                                    gene_ind = len(new_gene_to_ind)                                
                                    if gene_to_ind is not None:
                                        gene_ind += len(gene_to_ind)

                                    new_gene_to_ind[gene] = gene_ind
                                    genes.append(gene)
                                else:
                                    gene_ind = new_gene_to_ind[gene]

                            #store data for the later matrices
                            col.append(gene_set_ind)
                            row.append(gene_ind)
                            data.append(weight)
                        num_read += 1
                        cur_num_read += 1

                        #add at end or when have hit maximum
                        if len(data) >= max_num_entries_at_once:
                            log("Batching %d lines to save memory" % cur_num_read)
                            num_added, num_ignored = __add_to_X((data, row, col), genes, gene_sets, tag, skip_scale_factors=False)
                            if i == 0 and num_added + num_ignored == 0:
                                bail("--first-for-hyper was specified but first file had no gene sets")
                            #add gene set batches here
                            self.gene_set_batches = np.append(self.gene_set_batches, np.full(num_added, batches[i]))
                            self.gene_set_labels = np.append(self.gene_set_labels, np.full(num_added, labels[i]))
                            self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, np.full(num_ignored, labels[i]))
                            num_ignored_gene_sets[i] += num_ignored

                            #re-initialize things
                            genes = copy.copy(self.genes)
                            if self.genes_missing is not None:
                                genes += self.genes_missing
                            gene_to_ind = self._construct_map_to_ind(genes)
                            new_gene_to_ind = {}
                            gene_sets = []
                            data = []
                            row = []
                            col = []
                            num_read = 0
                            cur_num_read = 0
                            log("Continuing reading...")
                    #get the end if there are any

                    if already_seen > 0:
                        warn("Skipped second occurrence of %d repeated gene sets" % already_seen)

                    if len(data) > 0:
                        mat_info = (data, row, col)
                    else:
                        mat_info = None

                if mat_info is not None:

                    num_added, num_ignored = __add_to_X(mat_info, genes, gene_sets, tag, skip_scale_factors=False)
                    if i == 0 and num_added + num_ignored == 0:
                        bail("--first-for-hyper was specified but first file had no gene sets")
                    #add gene set batches here
                    self.gene_set_batches = np.append(self.gene_set_batches, np.full(num_added, batches[i]))
                    self.gene_set_labels = np.append(self.gene_set_labels, np.full(num_added, labels[i]))
                    self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, np.full(num_ignored, labels[i]))
                    num_ignored_gene_sets[i] += num_ignored

            log("Ignored %d gene sets due to too few genes" % num_too_small, DEBUG)

        if ignored_for_fraction_inc > 0:
            log("Ignored %d gene sets due to too small a fraction of anchor genes" % ignored_for_fraction_inc, DEBUG)

        if self.X_orig is None or self.X_orig.shape[1] == 0:
            log("No gene sets to analyze; returning")
            return

        if self.total_qc_metrics is not None:
            total_qc_metrics = self.total_qc_metrics
            if self.total_qc_metrics_ignored is not None:
                total_qc_metrics = np.vstack((self.total_qc_metrics, self.total_qc_metrics_ignored))

            self.total_qc_metrics = (self.total_qc_metrics - np.mean(total_qc_metrics, axis=0)) / np.std(total_qc_metrics, axis=0)
            if self.total_qc_metrics_ignored is not None:
                self.total_qc_metrics_ignored = (self.total_qc_metrics_ignored - np.mean(total_qc_metrics, axis=0)) / np.std(total_qc_metrics, axis=0)

        if self.mean_qc_metrics is not None:
            mean_qc_metrics = np.append(self.mean_qc_metrics, self.mean_qc_metrics_ignored if self.mean_qc_metrics_ignored is not None else [])
            self.mean_qc_metrics = (self.mean_qc_metrics - np.mean(mean_qc_metrics)) / np.std(mean_qc_metrics)
            if self.mean_qc_metrics_ignored is not None:
                self.mean_qc_metrics_ignored = (self.mean_qc_metrics_ignored - np.mean(mean_qc_metrics)) / np.std(mean_qc_metrics)

        if filter_gene_set_p is not None and (correct_betas_mean or correct_betas_var) and self.beta_tildes is not None:
            (self.beta_tildes, self.ses, self.z_scores, self.p_values, self.se_inflation_factors) = self._correct_beta_tildes(self.beta_tildes, self.ses, self.se_inflation_factors, self.total_qc_metrics, self.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, correct_ignored=True, fit=True)
            newly_below_p_mask = self.p_values <= filter_gene_set_p
            #ensure at least one
            if np.sum(newly_below_p_mask) == 0:
                newly_below_p_mask[np.argmin(self.p_values)] = True
            if np.sum(newly_below_p_mask) != len(newly_below_p_mask):
                log("Ignoring %d gene sets whose p-value increased after adjusting betas (kept %d)" % (np.sum(~newly_below_p_mask), np.sum(newly_below_p_mask)))
                self.subset_gene_sets(newly_below_p_mask, ignore_missing=True, keep_missing=False, skip_V=True)

        self._record_param("gene_set_prune_threshold", prune_gene_sets)
        self._record_param("gene_set_prune_deterinistically", prune_deterministically)

        if self.p_values is not None and max_num_gene_sets_initial is not None:

            if max_num_gene_sets_initial > 0 and max_num_gene_sets_initial < len(self.p_values):
                p_value_filter = np.partition(self.p_values, max_num_gene_sets_initial - 1)[max_num_gene_sets_initial - 1]
                log("Keeping only %d most significant gene sets due to --max-num-gene-sets-initial" % max_num_gene_sets_initial)
                self.subset_gene_sets(self.p_values <= p_value_filter, ignore_missing=True, keep_missing=False, skip_V=True)            

        if not skip_betas:
            self._prune_gene_sets(prune_gene_sets, prune_deterministically=prune_deterministically, keep_missing=False, ignore_missing=True, skip_V=True)

        #if permute_gene_sets:
        #    #assume these are due to some error in permutation
        #    min_allowed_p = 0.05 / (len(self.gene_sets) + len(self.gene_sets_ignored) if self.gene_sets_ignored is not None else 0)
        #    self.subset_gene_sets(self.p_values >= min_allowed_p, ignore_missing=True, keep_missing=False, skip_V=True)

        #if these were not set previously, use the initial values
        if self.p is None:
            self.set_p(initial_p)
        if self.sigma_power is None:
            self.set_sigma(self.sigma2, sigma_power)
        fixed_sigma_cond = False
        if self.sigma2 is None:
            if initial_sigma2_cond is not None:
                #if they specify cond sigma, we set the actual sigma (cond * p) and adjust for scale factors
                if not update_hyper_sigma:
                    fixed_sigma_cond = True
                self.set_sigma(self.p * initial_sigma2_cond, self.sigma_power)
            else:
                self.set_sigma(initial_sigma2, self.sigma_power)

        if sigma_soft_threshold_95 is not None and sigma_soft_threshold_5 is not None:
            if sigma_soft_threshold_95 < 0 or sigma_soft_threshold_5 < 0:
                warn("Ignoring sigma soft thresholding since both are not positive")
            else:
                #this will map scale factor to 
                frac_95 = float(sigma_soft_threshold_95) / len(self.genes)
                x1 = np.sqrt(frac_95 * (1 - frac_95))
                y1 = 0.95

                frac_5 = float(sigma_soft_threshold_5) / len(self.genes)
                x2 = np.sqrt(frac_5 * (1 - frac_5))
                y2 = 0.05
                L = 1

                if x2 < x1:
                    warn("--sigma-threshold-5 (%.3g) is less than --sigma-threshold-95 (%.3g); this is the opposite of what you usually want as it will threshold smaller gene sets rather than larger ones")

                self.sigma_threshold_k = -(np.log(1/y2 - L) - np.log(1/y1 - 1))/(x2-x1)
                self.sigma_threshold_xo = (x1 * np.log(1/y2 - L) - x2 * np.log(1/y1 - L)) / (np.log(1/y2 - L) - np.log(1/y1 - L))

                #self.sigma_threshold_xo = (x1 * np.log(L / y2 - 1) - x2 * np.log(L / y1 - 1)) / (np.log(L / y2 - 1) - np.log(L / y1 - 1))
                #self.sigma_threshold_k = -np.log(L / y2 - 1)/ (x2 - self.sigma_threshold_xo)

                log("Thresholding sigma with k=%.3g, xo=%.3g" % (self.sigma_threshold_k, self.sigma_threshold_xo))

        if not skip_betas and self.p_values is not None and (update_hyper_p or update_hyper_sigma) and len(self.gene_set_batches) > 0:

            #now learn the hyper values
            assert(self.gene_set_batches[0] is not None)
            #first order the unique batches; batches has one value per file but we need info one per unique batch
            ordered_batches = [self.gene_set_batches[0]] + list(set([x for x in self.gene_set_batches if not x == self.gene_set_batches[0]]))
            #get the total number of ignored genes per batch
            batches_num_ignored = {}
            for i in range(len(batches)):
                if batches[i] not in batches_num_ignored:
                    batches_num_ignored[batches[i]] = 0
                batches_num_ignored[batches[i]] += num_ignored_gene_sets[i]

            self.ps = np.full(len(self.gene_set_batches), np.nan)
            self.sigma2s = np.full(len(self.gene_set_batches), np.nan)

            #none learns from first; rest learn from within themselves
            first_p = None
            for ordered_batch_ind in range(len(ordered_batches)):

                if ordered_batches[ordered_batch_ind] is None:
                    #we'll be drawing this from the first
                    assert(first_for_hyper)
                    continue

                gene_sets_in_batch_mask = (self.gene_set_batches == ordered_batches[ordered_batch_ind])

                if ordered_batch_ind > 0 and np.sum(gene_sets_in_batch_mask) + batches_num_ignored[ordered_batches[ordered_batch_ind]] < 100:
                    log("Skipping learning hyper for batch %s since not enough gene sets" % (ordered_batches[ordered_batch_ind]))
                    continue

                #right now the way to pass these is to set member variables, so we save current and set
                orig_ps = self.ps
                orig_sigma2s = self.sigma2s
                #there are always none for running betas here
                self.ps = None
                self.sigma2s = None


                #orig_p = self.p
                #orig_sigma2 = self.sigma2
                #orig_sigma_power = self.sigma_power


                if np.sum(gene_sets_in_batch_mask) > self.batch_size:
                    V = None
                else:
                    V = self._calculate_V_internal(self.X_orig[:,gene_sets_in_batch_mask], self.y_corr_cholesky, self.mean_shifts[gene_sets_in_batch_mask], self.scale_factors[gene_sets_in_batch_mask])

                #run non_inf_betas
                #only add psuedo counts for large values
                num_p_pseudo = min(1, np.sum(gene_sets_in_batch_mask) / 1000)

                #adjust sigma means keep sigma/p constant (thereby adjusting unconditional variance=sigma)
                #if it is the first batch and first_for_hyper, we do not want to adjust the sigma
                #similarly, if it is not first_for_hyper, we do not want to adjust the sigma
                #we will learn it (if requested), but if not requested we assume that the specified sigma is the correct *UNCONDITIONAL* variance
                #thus, we will learn p subject to this constraint on total variance
                #after the first batch, however, when doing first_for_hyper, we will adjust sigma to keep the sigma/p fixed
                cur_update_hyper_p = update_hyper_p
                cur_update_hyper_sigma = update_hyper_sigma
                adjust_hyper_sigma_p = False
                if (first_for_sigma_cond and ordered_batch_ind > 0) or fixed_sigma_cond:
                    adjust_hyper_sigma_p = True
                    if cur_update_hyper_p:
                        cur_update_hyper_sigma = False
                Y_to_use = self.Y_for_regression
                Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))

                (betas, avg_postp) = self._calculate_non_inf_betas(initial_p=None, beta_tildes=self.beta_tildes[gene_sets_in_batch_mask], ses=self.ses[gene_sets_in_batch_mask], V=V, X_orig=self.X_orig[:,gene_sets_in_batch_mask], scale_factors=self.scale_factors[gene_sets_in_batch_mask], mean_shifts=self.mean_shifts[gene_sets_in_batch_mask], is_dense_gene_set=self.is_dense_gene_set[gene_sets_in_batch_mask], ps=None, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=False, update_hyper_sigma=cur_update_hyper_sigma, update_hyper_p=cur_update_hyper_p, adjust_hyper_sigma_p=adjust_hyper_sigma_p, sigma_num_devs_to_top=sigma_num_devs_to_top, p_noninf_inflate=p_noninf_inflate, num_p_pseudo=num_p_pseudo, num_missing_gene_sets=batches_num_ignored[ordered_batches[ordered_batch_ind]], sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, betas_trace_gene_sets=[self.gene_sets[j] for j in range(len(self.gene_sets)) if gene_sets_in_batch_mask[j]])

                #now save and restore
                computed_p = self.p
                computed_sigma2 = self.sigma2
                computed_sigma_power = self.sigma_power

                #don't reset 
                #if not first_for_sigma_cond and ordered_batch_ind > 0:
                #    self.set_p(orig_p)
                #    self.set_sigma(orig_sigma2, orig_sigma_power)

                self.ps = orig_ps
                self.sigma2s = orig_sigma2s

                log("Learned p=%.4g, sigma2=%.4g (sigma2/p=%.4g)" % (computed_p, computed_sigma2, computed_sigma2/computed_p))
                self._record_params({"p": computed_p, "sigma2": computed_sigma2, "sigma2_cond": computed_sigma2/computed_p, "sigma_power": computed_sigma_power, "sigma_threshold_k": self.sigma_threshold_k, "sigma_threshold_xo": self.sigma_threshold_xo})

                if first_p is None:
                    first_p = computed_p
                elif first_max_p_for_hyper and computed_p > first_p:
                    #keep sigma/first_p = sigma/computed_p
                    computed_sigma2 = computed_sigma2 / computed_p * first_p
                    computed_p = first_p

                self.ps[gene_sets_in_batch_mask] = computed_p
                self.sigma2s[gene_sets_in_batch_mask] = computed_sigma2

            #take care of the missing ps

            assert(len(self.ps) > 0 and not np.isnan(self.ps[0]))
            assert(len(self.sigma2s) > 0 and not np.isnan(self.sigma2s[0]))

            if first_for_hyper:
                self.ps[np.isnan(self.ps)] = self.ps[0]
                self.sigma2s[np.isnan(self.sigma2s)] = self.sigma2s[0]
            else:
                #this should only occur if the gene sets were too small
                self.ps[np.isnan(self.ps)] = np.mean(self.ps[~np.isnan(self.ps)])
                self.sigma2s[np.isnan(self.sigma2s)] = np.mean(self.sigma2s[~np.isnan(self.sigma2s)])

            self.set_p(np.mean(self.ps))
            self.set_sigma(np.mean(self.sigma2s), self.sigma_power)

            #if shared_sigma_cond:
            #    #we want sigma2/self.p2 to be constant
            #    self.sigma2s = self.sigma2 * self.ps / self.p

        if filter_gene_set_p is not None and increase_filter_gene_set_p is not None and self.p_values is not None and self.p_values_ignored is not None:
            #since we required each batch to have increase_filter_gene_set_p, maybe we need to reduce
            if float(len(self.p_values)) / (len(self.p_values) + len(self.p_values_ignored)) > increase_filter_gene_set_p:
                #choose a potentially more strict threshold
                #want keep_frac * len(self.p_values) / (len(self.p_values) + len(self.p_values_ignored)) = filter_gene_set_p
                keep_frac = increase_filter_gene_set_p * float(len(self.p_values) + len(self.p_values_ignored)) / len(self.p_values)
                p_from_quantile = np.quantile(self.p_values, keep_frac)
                if p_from_quantile > filter_gene_set_p:
                    overcorrect_ignore = self.p_values > p_from_quantile
                    if np.sum(overcorrect_ignore) > 0:
                        overcorrect_mask = ~overcorrect_ignore
                        self._record_param("adjusted_filter_gene_set_p", p_from_quantile)
                        log("Ignoring %d gene sets due to p > %.3g (overaggressive adjustment of p-value filters; kept %d)" % (np.sum(overcorrect_ignore), p_from_quantile, np.sum(overcorrect_mask)))
                        self.subset_gene_sets(overcorrect_mask, ignore_missing=True, keep_missing=False, skip_V=True)

        #do another check of min_gene_set_size in case we converted some gene sets with weights
        if self.X_orig is not None:
            col_sums = self.get_col_sums(self.X_orig, num_nonzero=True)
            size_ignore = col_sums < min_gene_set_size

            if np.sum(size_ignore) > 0:
                size_mask = ~size_ignore
                log("Ignoring %d gene sets due to too few genes (kept %d)" % (np.sum(size_ignore), np.sum(size_mask)))
                self.subset_gene_sets(size_mask, keep_missing=False, skip_V=True)

            col_sums = self.get_col_sums(self.X_orig, num_nonzero=True)
            size_ignore = col_sums > max_gene_set_size
            if np.sum(size_ignore) > 0:
                size_mask = ~size_ignore
                log("Ignoring %d gene sets due to too many genes (kept %d)" % (np.sum(size_ignore), np.sum(size_mask)))
                self.subset_gene_sets(size_mask, keep_missing=False, skip_V=True)

            if self.total_qc_metrics is not None and filter_gene_set_metric_z:
                filter_mask = np.abs(self.mean_qc_metrics) < filter_gene_set_metric_z
                filter_ignore = ~filter_mask
                log("Ignoring %d gene sets due to QC metric filters (kept %d)" % (np.sum(filter_ignore), np.sum(filter_mask)))
                self.subset_gene_sets(filter_mask, keep_missing=False, ignore_missing=True, skip_V=True)

                #self.total_qc_metrics = np.vstack((self.mean_qc_metrics, np.ones(len(self.mean_qc_metrics)))).T
                #self.total_qc_metrics_ignored = np.vstack((self.mean_qc_metrics_ignored, np.ones(len(self.mean_qc_metrics_ignored)))).T

        if self.p_values is not None:
            sort_rank = self.p_values
        else:
            sort_rank = np.arange(len(self.gene_sets))
        if not skip_betas and self.p_values is not None and filter_gene_set_p < 1:
            #remove those that have uncorrected beta equal to zero
            (betas, avg_postp) = self._calculate_non_inf_betas(initial_p=None, assume_independent=True, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=False, update_hyper_sigma=False, update_hyper_p=False, adjust_hyper_sigma_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas)

            beta_ignore = betas == 0
            beta_mask = ~beta_ignore
            if np.sum(beta_mask) > 0:
                log("Ignoring %d gene sets due to zero uncorrected betas (kept %d)" % (np.sum(beta_ignore), np.sum(beta_mask)))
                self.subset_gene_sets(beta_mask, keep_missing=False, ignore_missing=True, skip_V=True)
            else:
                log("Keeping %d gene sets with zero uncorrected betas to avoid having none" % (np.sum(beta_ignore)))

            sort_rank = -np.abs(betas[beta_mask])

        if max_num_gene_sets is not None and len(self.gene_sets) > max_num_gene_sets and max_num_gene_sets > 0:
            log("Current %d gene sets is greater than maximum specified %d; reducing using pruning + small beta removal" % (len(self.gene_sets), max_num_gene_sets), DEBUG)
            gene_set_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig, mean_shifts=self.mean_shifts, scale_factors=self.scale_factors, sort_values=sort_rank, stop_at=max_num_gene_sets)
            keep_mask = np.full(len(self.gene_sets), False)
            for gene_set_mask in gene_set_masks:
                keep_mask[gene_set_mask] = True
                log("Adding %d relatively uncorrelated gene sets (total now %d)" % (np.sum(gene_set_mask), np.sum(keep_mask)), TRACE)
                if np.sum(keep_mask) > max_num_gene_sets:
                    break
            if np.sum(keep_mask) > max_num_gene_sets:
                threshold_value = sorted(sort_rank[keep_mask])[max_num_gene_sets - 1]
                keep_mask[sort_rank > threshold_value] = False
            if np.sum(~keep_mask) > 0:
                self.subset_gene_sets(keep_mask, keep_missing=False, ignore_missing=True, skip_V=True)

        self._record_param("num_gene_sets_read", len(self.gene_sets))
        self._record_param("num_genes_read", len(self.genes))

        log("Read %d gene sets and %d genes" % (len(self.gene_sets), len(self.genes)))

    #this reads a V matrix directly from a file
    #it does not initialize an X matrix; if the X-matrix is needed, read_X should be used instead
    def read_V(self, V_in):

        log("Reading V from --V-in file %s" % V_in, INFO)
        with open(V_in) as V_fh:
            header = V_fh.readline().strip()
        if len(header) == 0 or header[0] != "#":
            bail("First line of --V-in must be proceeded by #")
        header = header.lstrip("# \t")
        gene_sets = header.split()
        if len(gene_sets) < 1:
            bail("First line of --X-in must contain list of gene sets")

        gene_set_to_ind = self._construct_map_to_ind(gene_sets)
        V = np.genfromtxt(V_in, skip_header=1)
        if V.shape[0] != V.shape[1] or V.shape[0] != len(gene_sets):
            bail("V matrix dimensions %s do not match number of gene sets in header line (%s)" % (V.shape, len(gene_sets)))

        if self.gene_sets is None:
            self.gene_sets = gene_sets
            self.gene_set_to_ind = gene_set_to_ind
        else:
            #first remove everything from V that is not in gene sets previously
            subset_mask = np.array([(x in self.gene_set_to_ind) for x in gene_sets])
            if sum(subset_mask) != len(subset_mask):
                warn("Excluding %s values from previously loaded files because absent from --V-in file" % (len(subset_mask) - sum(subset_mask)))
                V = V[subset_mask,:][:,subset_mask]
                self.gene_sets = list(itertools.compress(self.gene_sets, subset_mask))
                self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)
            #now remove everything from the other files that are not in V
            old_subset_mask = np.array([(x in gene_set_to_ind) for x in self.gene_sets])
            if sum(old_subset_mask) != len(old_subset_mask):
                warn("Excluding %s values from --V-in file because absent from previously loaded files" % (len(old_subset_mask) - sum(old_subset_mask)))
                self.subset_gene_sets(old_subset_mask, keep_missing=False, skip_V=True)
        return V


    def write_V(self, V_out):
        if self.X_orig is not None:
            V = self._get_V()
            log("Writing V matrix to %s" % V_out, INFO)
            np.savetxt(V_out, V, delimiter='\t', fmt="%.2g", comments="#", header="%s" % ("\t".join(self.gene_sets)))
        else:
            warn("V has not been initialized; skipping writing")

    def write_Xd(self, X_out):
        if self.X_orig is not None:
            log("Writing X matrix to %s" % X_out, INFO)
            #FIXME: get_orig_X
            np.savetxt(X_out, self.X_orig.toarray(), delimiter='\t', fmt="%.3g", comments="#", header="%s" % ("%s\n#%s" % ("\t".join(self.gene_sets), "\t".join(self.genes))))
        else:
            warn("X has not been initialized; skipping writing")

    def write_X(self, X_out):
        if self.genes is None or self.X_orig is None or self.gene_sets is None:
            return
            warn("X has not been initialized; skipping writing")
            return

        log("Writing X sparse matrix to %s" % X_out, INFO)

        with open_gz(X_out, 'w') as output_fh:

            for j in range(len(self.gene_sets)):
                line = self.gene_sets[j]
                nonzero_inds = self.X_orig[:,j].nonzero()[0]
                non_unity = np.sum(self.X_orig[nonzero_inds,j] == 1) < len(nonzero_inds)
                for i in nonzero_inds:
                    if non_unity:
                        line = "%s\t%s:%.2g" % (line, self.genes[i], self.X_orig[i,j])
                    else:
                        line = "%s\t%s" % (line, self.genes[i])

                output_fh.write("%s\n" % line)

    def calculate_huge_scores_gwas(self, gwas_in, gwas_chrom_col=None, gwas_pos_col=None, gwas_p_col=None, gene_loc_file=None, hold_out_chrom=None, exons_loc_file=None, gwas_beta_col=None, gwas_se_col=None, gwas_n_col=None, gwas_n=None, gwas_freq_col=None, gwas_filter_col=None, gwas_filter_value=None, gwas_locus_col=None, gwas_ignore_p_threshold=None, gwas_units=None, gwas_low_p=5e-8, gwas_high_p=1e-2, gwas_low_p_posterior=0.98, gwas_high_p_posterior=0.001, detect_low_power=None, detect_high_power=None, detect_adjust_huge=False, learn_window=False, closest_gene_prob=0.7, max_closest_gene_prob=0.9, scale_raw_closest_gene=True, cap_raw_closest_gene=False, cap_region_posterior=True, scale_region_posterior=False, phantom_region_posterior=False, allow_evidence_of_absence=False, correct_huge=True, max_signal_p=1e-5, signal_window_size=250000, signal_min_sep=100000, signal_max_logp_ratio=None, credible_set_span=25000, max_closest_gene_dist=2.5e5, min_n_ratio=0.5, max_clump_ld=0.2, min_var_posterior=0.01, s2g_in=None, s2g_chrom_col=None, s2g_pos_col=None, s2g_gene_col=None, s2g_prob_col=None, s2g_normalize_values=None, credible_sets_in=None, credible_sets_id_col=None, credible_sets_chrom_col=None, credible_sets_pos_col=None, credible_sets_ppa_col=None, **kwargs):
        if gwas_in is None:
            bail("Require --gwas-in for this operation")
        if gene_loc_file is None:
            bail("Require --gene-loc-file for this operation")

        if credible_sets_in is not None:
            if credible_sets_chrom_col is None or credible_sets_pos_col is None:
                bail("Need --credible-set-chrom-col and --credible-set-pos-col")

        if signal_window_size < 2 * signal_min_sep:
            signal_window_size = 2 * signal_min_sep

        if signal_max_logp_ratio is not None:
            if signal_max_logp_ratio > 1:
                warn("Thresholding --signal-max-logp-ratio at 1")
                signal_max_logp_ratio = 1

        self._record_params({"gwas_low_p": gwas_low_p, "gwas_high_p": gwas_high_p, "gwas_low_p_posterior": gwas_low_p_posterior, "gwas_high_p_posterior": gwas_high_p_posterior, "detect_low_power": detect_low_power, "detect_high_power": detect_high_power, "detect_adjust_huge": detect_adjust_huge, "closest_gene_prob": closest_gene_prob, "max_closest_gene_prob": max_closest_gene_prob, "scale_raw_closest_gene": scale_raw_closest_gene, "cap_raw_closest_gene": cap_raw_closest_gene, "cap_region_posterior": cap_region_posterior, "scale_region_posterior": scale_region_posterior, "max_signal_p": max_signal_p, "signal_window_size": signal_window_size, "signal_min_sep": signal_min_sep, "max_closest_gene_dist": max_closest_gene_dist, "min_n_ratio": min_n_ratio})

        #see if need to determine
        if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
            need_columns = True
        else:
            has_se = gwas_se_col is not None or gwas_n_col is not None or gwas_n is not None
            if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and has_se) or (gwas_beta_col is not None and has_se):
                need_columns = False
            else:
                need_columns = True

        if need_columns:
            (possible_gene_id_cols, possible_var_id_cols, possible_chrom_cols, possible_pos_cols, possible_locus_cols, possible_p_cols, possible_beta_cols, possible_se_cols, possible_freq_cols, possible_n_cols, header) = self._determine_columns(gwas_in)

            #now recompute
            if gwas_pos_col is None:
                if len(possible_pos_cols) == 1:
                    gwas_pos_col = possible_pos_cols[0]
                    log("Using %s for position column; change with --gwas-pos-col if incorrect" % gwas_pos_col)
                else:
                    log("Could not determine position column from header %s; specify with --gwas-pos-col" % header)
            if gwas_chrom_col is None:
                if len(possible_chrom_cols) == 1:
                    gwas_chrom_col = possible_chrom_cols[0]
                    log("Using %s for chrom column; change with --gwas-chrom-col if incorrect" % gwas_chrom_col)
                else:
                    log("Could not determine chrom column from header %s; specify with --gwas-chrom-col" % header)
            if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
                if len(possible_locus_cols) == 1:
                    gwas_locus_col = possible_locus_cols[0]
                    log("Using %s for locus column; change with --gwas-locus-col if incorrect" % gwas_locus_col)
                else:
                    bail("Could not determine chrom and pos columns from header %s; specify with --gwas-chrom-col and --gwas-pos-col or with --gwas-locus-col" % header)

            if gwas_p_col is None:
                if len(possible_p_cols) == 1:
                    gwas_p_col = possible_p_cols[0]
                    log("Using %s for p column; change with --gwas-p-col if incorrect" % gwas_p_col)
                else:
                    log("Could not determine p column from header %s; if desired specify with --gwas-p-col" % header)
            if gwas_se_col is None:
                if len(possible_se_cols) == 1:
                    gwas_se_col = possible_se_cols[0]
                    log("Using %s for se column; change with --gwas-se-col if incorrect" % gwas_se_col)
                else:
                    log("Could not determine se column from header %s; if desired specify with --gwas-se-col" % header)
            if gwas_beta_col is None:
                if len(possible_beta_cols) == 1:
                    gwas_beta_col = possible_beta_cols[0]
                    log("Using %s for beta column; change with --gwas-beta-col if incorrect" % gwas_beta_col)
                else:
                    log("Could not determine beta column from header %s; if desired specify with --gwas-beta-col" % header)

            if gwas_n_col is None:
                if len(possible_n_cols) == 1:
                    gwas_n_col = possible_n_cols[0]
                    log("Using %s for N column; change with --gwas-n-col if incorrect" % gwas_n_col)
                else:
                    log("Could not determine N column from header %s; if desired specify with --gwas-n-col" % header)

            if gwas_freq_col is None:
                if len(possible_freq_cols) == 1:
                    gwas_freq_col = possible_freq_cols[0]
                    log("Using %s for freq column; change with --gwas-freq-col if incorrect" % gwas_freq_col)

            has_se = gwas_se_col is not None
            has_n = gwas_n_col is not None or gwas_n is not None
            if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and (has_se or has_n)) or (gwas_beta_col is not None and has_se):
                pass
            else:
                bail("Require information about p-value and se or N or beta, or beta and se; specify with --gwas-p-col, --gwas-beta-col, and --gwas-se-col")

            if options.debug_just_check_header:
                bail("Done checking headers")


        #use this to store the exons 
        class IntervalTree(object):
            __slots__ = ('interval_starts', 'interval_stops', 'left', 'right', 'center')
            def __init__(self, intervals, depth=16, minbucket=96, _extent=None, maxbucket=4096):
                depth -= 1
                if (depth == 0 or len(intervals) < minbucket) and len(intervals) > maxbucket:
                    self.interval_starts, self.interval_stops = zip(*intervals)
                    self.left = self.right = None
                    return

                left, right = _extent or (min(i[0] for i in intervals), max(i[1] for i in intervals))
                center = (left + right) / 2.0

                self.interval_starts = []
                self.interval_stops = []
                lefts, rights  = [], []

                for interval in intervals:
                    if interval[1] < center:
                        lefts.append(interval)
                    elif interval[0] > center:
                        rights.append(interval)
                    else: # overlapping.
                        self.interval_starts.append(interval[0])
                        self.interval_stops.append(interval[1])

                self.interval_starts = np.array(self.interval_starts)
                self.interval_stops = np.array(self.interval_stops)
                self.left   = lefts  and IntervalTree(lefts,  depth, minbucket, (left,  center)) or None
                self.right  = rights and IntervalTree(rights, depth, minbucket, (center, right)) or None
                self.center = center

            def find(self, start, stop, index_map=None):

                #find overlapping intervals for set of (start, stop) pairs
                #start is array of starting points
                #stop is array of corresponding stopping points
                #return list of two np arrays. First of passed in indices for which there is an overlap (indices can be repeated)
                #second a list of intervals that overlap
                """find all elements between (or overlapping) start and stop"""
                #array with rows equal to intervals length, columns equal to stop, true if intervals less than or equal to stop
                less_mask = np.less(self.interval_starts, stop[:,np.newaxis] + 1)
                #array with rows equal to intervals length, columns equal to stop, true if intervals greater than or equal to stop
                greater_mask = np.greater(self.interval_stops, start[:,np.newaxis] - 1)
                #interval x variant pos array with the intervals that overlap each variant pos
                overlapping_mask = np.logical_and(less_mask, greater_mask)

                #tuple of (overlapping interval indices, passed in start/stop index with the overlap)
                overlapping_where = np.where(overlapping_mask)

                overlapping_indices = (overlapping_where[0], self.interval_starts[overlapping_where[1]], self.interval_stops[overlapping_where[1]])
                #overlapping = [i for i in self.intervals if i[1] >= start and i[0] <= stop]

                start_less_mask = start <= self.center
                if self.left and np.any(start_less_mask):
                    left_overlapping_indices = self.left.find(start[start_less_mask], stop[start_less_mask], index_map=np.where(start_less_mask)[0])
                    overlapping_indices = (np.append(overlapping_indices[0], left_overlapping_indices[0]), np.append(overlapping_indices[1], left_overlapping_indices[1]), np.append(overlapping_indices[2], left_overlapping_indices[2]))

                stop_greater_mask = stop >= self.center
                if self.right and np.any(stop_greater_mask):
                    right_overlapping_indices = self.right.find(start[stop_greater_mask], stop[stop_greater_mask], index_map=np.where(stop_greater_mask)[0])
                    overlapping_indices = (np.append(overlapping_indices[0], right_overlapping_indices[0]), np.append(overlapping_indices[1], right_overlapping_indices[1]), np.append(overlapping_indices[2], right_overlapping_indices[2]))

                if index_map is not None and len(overlapping_indices[0]) > 0:
                    overlapping_indices = (index_map[overlapping_indices[0]], overlapping_indices[1], overlapping_indices[2])

                return overlapping_indices


        #store the gene locations
        log("Reading gene locations")
        (gene_chrom_name_pos, gene_to_chrom, gene_to_pos) = self._read_loc_file(gene_loc_file, hold_out_chrom=hold_out_chrom)

        for chrom in gene_chrom_name_pos:
            serialized_gene_info = []
            for gene in gene_chrom_name_pos[chrom]:
                for pos in gene_chrom_name_pos[chrom][gene]:
                    serialized_gene_info.append((gene,pos))
            gene_chrom_name_pos[chrom] = serialized_gene_info 

        chrom_to_interval_tree = None
        if exons_loc_file is not None:

            log("Reading exon locations")

            chrom_interval_to_gene = self._read_loc_file(exons_loc_file, return_intervals=True)
            chrom_to_interval_tree = {}
            for chrom in chrom_interval_to_gene:
                chrom_to_interval_tree[chrom] = IntervalTree(chrom_interval_to_gene[chrom].keys())

        (allelic_var_k, gwas_prior_odds) = self.compute_allelic_var_and_prior(gwas_high_p, gwas_high_p_posterior, gwas_low_p, gwas_low_p_posterior)
        #this stores the original values, in case we detect low or high power
        (allelic_var_k_detect, gwas_prior_odds_detect) = (allelic_var_k, gwas_prior_odds)
        separate_detect = False

        var_z_threshold = None
        var_p_threshold = None

        if min_var_posterior is not None and min_var_posterior > gwas_high_p_posterior:
            #var_log_bf + np.log(gwas_prior_odds) > np.log(min_var_posterior / (1 - min_var_posterior))
            #var_log_bf > np.log(min_var_posterior / (1 - min_var_posterior)) - np.log(gwas_prior_odds) = threshold
            #-np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k) > threshold
            #0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k) > threshold + np.log(np.sqrt(1 + allelic_var_k))
            #np.square(var_z) * allelic_var_k / (1 + allelic_var_k) > 2 * (threshold + np.log(np.sqrt(1 + allelic_var_k)))
            #np.square(var_z) * allelic_var_k > 2 * (1 + allelic_var_k) * (threshold + np.log(np.sqrt(1 + allelic_var_k)))
            log_bf_threshold = np.log(min_var_posterior / (1 - min_var_posterior)) - np.log(gwas_prior_odds) + np.log(np.sqrt(1 + allelic_var_k))

            if log_bf_threshold > 0:
                var_z_threshold = np.sqrt(2 * (1 + allelic_var_k) * (log_bf_threshold) / allelic_var_k)
                var_p_threshold = 2*scipy.stats.norm.cdf(-np.abs(var_z_threshold))
                log("Keeping only variants with p < %.4g" % var_p_threshold)
        else:
            var_p_threshold = gwas_high_p_posterior
            var_z_threshold = np.abs(scipy.stats.norm.ppf(var_p_threshold / 2))


        log("Reading --gwas-in file %s" % gwas_in, INFO)

        with open_gz(gwas_in) as gwas_fh:

            split_char = None
            header_line = gwas_fh.readline().strip()
            if '\t' in header_line:
                split_char = '\t'
            header_cols = header_line.split(split_char)
            header_cols = [x for x in header_cols if x != ""]

            chrom_col = None
            pos_col = None
            locus_col = None
            if gwas_chrom_col is not None and gwas_pos_col is not None:
                chrom_col = self._get_col(gwas_chrom_col, header_cols)
                pos_col = self._get_col(gwas_pos_col, header_cols)
            else:
                locus_col = self._get_col(gwas_locus_col, header_cols)

            p_col = None
            if gwas_p_col is not None:
                p_col = self._get_col(gwas_p_col, header_cols)

            beta_col = None
            if gwas_beta_col is not None:
                beta_col = self._get_col(gwas_beta_col, header_cols)

            n_col = None
            se_col = None
            if gwas_n_col is not None:
                n_col = self._get_col(gwas_n_col, header_cols)
            if gwas_se_col is not None:
                se_col = self._get_col(gwas_se_col, header_cols)

            freq_col = None
            if gwas_freq_col is not None:
                freq_col = self._get_col(gwas_freq_col, header_cols)

            filter_col = None
            if gwas_filter_col is not None:
                filter_col = self._get_col(gwas_filter_col, header_cols)

            chrom_pos_p_beta_se_freq = {}
            seen_chrom_pos = {}

            if (chrom_col is None or pos_col is None) and locus_col is None:
                bail("Operation requires --gwas-chrom-col and --gwas-pos-col or --gwas-locus-col")

            #read in the gwas associations
            total_num_vars = 0

            mean_n = 0

            warned_pos = False
            warned_stats = False

            not_enough_info = 0
            for line in gwas_fh:

                #TODO: allow a separate snp-loc file to be used

                cols = line.strip().split(split_char)
                if (chrom_col is not None and chrom_col > len(cols)) or (pos_col is not None and pos_col > len(cols)) or (locus_col is not None and locus_col > len(cols)) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)) or (n_col is not None and n_col > len(cols)) or (freq_col is not None and freq_col > len(cols) or (filter_col is not None and filter_col > len(cols))):
                    warn("Skipping line due to too few columns: %s" % line)
                    continue

                if filter_col is not None and gwas_filter_value is not None and cols[filter_col] != gwas_filter_value:
                    continue

                if chrom_col is not None and pos_col is not None:
                    chrom = cols[chrom_col]
                    pos = cols[pos_col]
                else:
                    locus = cols[locus_col]
                    locus_tokens = None
                    for locus_delim in [":", "_"]:
                        if locus_delim in locus:
                            locus_tokens = locus.split(locus_delim)
                            break
                    if locus_tokens is None or len(locus_tokens) <= 2:
                        bail("Could not split locus %s on either : or _" % locus)
                    chrom = locus_tokens[0]
                    pos = locus_tokens[1]

                chrom = self._clean_chrom(chrom)
                if hold_out_chrom is not None and chrom == hold_out_chrom:
                    continue
                try:
                    pos = int(pos)
                except ValueError:
                    if not warned_pos:
                        warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                        warned_pos = True
                    continue

                p = None
                if p_col is not None:
                    try:
                        p = float(cols[p_col])
                    except ValueError:
                        if not cols[p_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible p value %s" % (cols[p_col]))
                                warned_stats = True
                        p = None

                    if p is not None:
                        min_p = 1e-250
                        if p < min_p:
                            p = min_p

                        if p <= 0 or p > 1:
                            if not warned_stats:
                                warn("Skipping invalid p value %s" % (p))
                                warned_stats = True
                            p = None

                        if gwas_ignore_p_threshold is not None and p > gwas_ignore_p_threshold:
                            continue

                beta = None
                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                    except ValueError:
                        if not cols[beta_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible beta value %s" % (cols[beta_col]))
                                warned_stats = True
                        beta = None

                se = None
                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible se value %s" % (cols[se_col]))
                                warned_stats = True
                        se = None

                if se is None:
                    if n_col is not None:
                        try:
                            n = float(cols[n_col])
                            if n <= 0:
                                if not warned_stats:
                                    warn("Skipping invalid N value %s" % (n))
                                    warned_stats = True
                                n = None

                        except ValueError:
                            if not cols[n_col] == "NA":
                                if not warned_stats:
                                    warn("Skipping unconvertible n value %s" % (cols[n_col]))
                                    warned_stats = True
                            n = None

                        if n is not None:
                            se = 1 / np.sqrt(n)

                    elif gwas_n is not None:
                        if gwas_n <= 0:
                            bail("Invalid gwas-n value: %s" % (gwas_n))

                        n = gwas_n
                        se = 1 / np.sqrt(n)


                #make sure have two of the three
                if sum((p is not None, se is not None, beta is not None)) < 2:
                    not_enough_info += 1
                    continue

                if var_z_threshold is not None:
                    if p is not None:
                        if p > var_p_threshold:
                            continue
                    else:
                        if se == 0:
                            continue
                        z = np.abs(beta / se)
                        if z < var_z_threshold:
                            continue

                freq = None
                if freq_col is not None:
                    try:
                        freq = float(cols[freq_col])
                        if freq > 1 or freq < 0:
                            warn("Skipping invalid freq value %s" % freq)
                            freq = None
                    except ValueError:
                        if not cols[freq_col] == "NA":
                            warn("Skipping unconvertible n value %s" % (cols[freq_col]))
                        freq = None


                if chrom not in chrom_pos_p_beta_se_freq:
                    chrom_pos_p_beta_se_freq[chrom] = []

                chrom_pos_p_beta_se_freq[chrom].append((pos, p, beta, se, freq))
                if chrom not in seen_chrom_pos:
                    seen_chrom_pos[chrom] = set()
                seen_chrom_pos[chrom].add(pos)
                total_num_vars += 1

            if not_enough_info > 0:
                warn("Skipped %d variants due to not enough information" % (not_enough_info))

            log("Read in %d variants" % total_num_vars)
            chrom_pos_to_gene_prob = None
            if s2g_in is not None:
                chrom_pos_to_gene_prob = {}

                log("Reading --s2g-in file %s" % s2g_in, INFO)

                #see if need to determine
                if s2g_pos_col is None or s2g_chrom_col is None or s2g_gene_col is None:
                    (possible_s2g_gene_cols, possible_s2g_var_id_cols, possible_s2g_chrom_cols, possible_s2g_pos_cols, possible_s2g_locus_cols, possible_s2g_p_cols, possible_s2g_beta_cols, possible_s2g_se_cols, possible_s2g_freq_cols, possible_s2g_n_cols) = self._determine_columns(s2g_in)

                    if s2g_pos_col is None:
                        if len(possible_s2g_pos_cols) == 1:
                            s2g_pos_col = possible_s2g_pos_cols[0]
                            log("Using %s for position column; change with --s2g-pos-col if incorrect" % s2g_pos_col)
                        else:
                            bail("Could not determine position column; specify with --s2g-pos-col")
                    if s2g_chrom_col is None:
                        if len(possible_s2g_chrom_cols) == 1:
                            s2g_chrom_col = possible_s2g_chrom_cols[0]
                            log("Using %s for chromition column; change with --s2g-chrom-col if incorrect" % s2g_chrom_col)
                        else:
                            bail("Could not determine chrom column; specify with --s2g-chrom-col")
                    if s2g_gene_col is None:
                        if len(possible_s2g_gene_cols) == 1:
                            s2g_gene_col = possible_s2g_gene_cols[0]
                            log("Using %s for geneition column; change with --s2g-gene-col if incorrect" % s2g_gene_col)
                        else:
                            bail("Could not determine gene column; specify with --s2g-gene-col")

                with open_gz(s2g_in) as s2g_fh:
                    header_cols = s2g_fh.readline().strip().split()
                    chrom_col = self._get_col(s2g_chrom_col, header_cols)
                    pos_col = self._get_col(s2g_pos_col, header_cols)
                    gene_col = self._get_col(s2g_gene_col, header_cols)
                    prob_col = None
                    if s2g_prob_col is not None:
                        prob_col = self._get_col(s2g_prob_col, header_cols)

                    for line in s2g_fh:

                        cols = line.strip().split()
                        if chrom_col > len(cols) or pos_col > len(cols) or gene_col > len(cols) or (prob_col is not None and prob_col > len(cols)):
                            warn("Skipping due to too few columns in line: %s" % line)
                            continue

                        chrom = self._clean_chrom(cols[chrom_col])
                        if hold_out_chrom is not None and chrom == hold_out_chrom:
                            continue

                        try:
                            pos = int(cols[pos_col])
                        except ValueError:
                            warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                            continue
                        gene = cols[gene_col]

                        if self.gene_label_map is not None and gene in self.gene_label_map:
                            gene = self.gene_label_map[gene]

                        max_s2g_prob=0.95
                        prob = max_s2g_prob
                        if prob_col is not None:
                            try:
                                prob = float(cols[prob_col])

                            except ValueError:
                                warn("Skipping unconvertible prob value %s" % (cols[prob_col]))
                                continue
                        if prob > max_s2g_prob:
                            prob = max_s2g_prob

                        if chrom in seen_chrom_pos and pos in seen_chrom_pos[chrom]:
                            if chrom not in chrom_pos_to_gene_prob:
                                chrom_pos_to_gene_prob[chrom] = {}
                            if pos not in chrom_pos_to_gene_prob[chrom]:
                                chrom_pos_to_gene_prob[chrom][pos] = []
                            chrom_pos_to_gene_prob[chrom][pos].append((gene, prob))

                    if s2g_normalize_values is not None:

                        for chrom in chrom_pos_to_gene_prob:
                            for pos in chrom_pos_to_gene_prob[chrom]:
                                prob_sum = sum([x[1] for x in chrom_pos_to_gene_prob[chrom][pos]])
                                if prob_sum > 0:
                                    norm_factor = s2g_normalize_values / prob_sum
                                    chrom_pos_to_gene_prob[chrom][pos] = [(x[0], x[1] * norm_factor) for x in chrom_pos_to_gene_prob[chrom][pos]]

            added_chrom_pos = {}
            input_credible_set_info = {}
            if credible_sets_in is not None:

                log("Reading --credible-sets-in file %s" % credible_sets_in, INFO)

                #see if need to determine
                if credible_sets_pos_col is None or credible_sets_chrom_col is None:
                    (_, _, possible_credible_sets_chrom_cols, possible_credible_sets_pos_cols, _, _, _, _, _, _, header) = self._determine_columns(credible_sets_in)

                    if credible_sets_pos_col is None:
                        if len(possible_credible_sets_pos_cols) == 1:
                            credible_sets_pos_col = possible_credible_sets_pos_cols[0]
                            log("Using %s for position column; change with --credible-sets-pos-col if incorrect" % credible_sets_pos_col)
                        else:
                            bail("Could not determine position column; specify with --credible-sets-pos-col")
                    if credible_sets_chrom_col is None:
                        if len(possible_credible_sets_chrom_cols) == 1:
                            credible_sets_chrom_col = possible_credible_sets_chrom_cols[0]
                            log("Using %s for chromition column; change with --credible-sets-chrom-col if incorrect" % credible_sets_chrom_col)
                        else:
                            bail("Could not determine chrom column; specify with --credible-sets-chrom-col")

                with open_gz(credible_sets_in) as credible_sets_fh:
                    header_cols = credible_sets_fh.readline().strip().split()
                    chrom_col = self._get_col(credible_sets_chrom_col, header_cols)
                    pos_col = self._get_col(credible_sets_pos_col, header_cols)
                    id_col = None
                    if credible_sets_id_col is not None:
                        id_col = self._get_col(credible_sets_id_col, header_cols)
                    ppa_col = None
                    if credible_sets_ppa_col is not None:
                        ppa_col = self._get_col(credible_sets_ppa_col, header_cols)

                    for line in credible_sets_fh:

                        cols = line.strip().split()
                        if (id_col is not None and id_col > len(cols)) or (chrom_col is not None and chrom_col > len(cols)) or (pos_col is not None and pos_col > len(cols)) or (ppa_col is not None and ppa_col > len(cols)):
                            warn("Skipping due to too few columns in line: %s" % line)
                            continue

                        chrom = self._clean_chrom(cols[chrom_col])

                        if hold_out_chrom is not None and chrom == hold_out_chrom:
                            continue

                        try:
                            pos = int(cols[pos_col])
                        except ValueError:
                            warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                            continue

                        if id_col is not None:
                            cs_id = cols[id_col]
                        else:
                            cs_id = "%s:%s" % (chrom, pos)

                        ppa = None
                        if ppa_col is not None:
                            try:
                                ppa = float(cols[ppa_col])
                                if ppa > 1:
                                    ppa = 0.99
                                elif ppa < 0:
                                    ppa = 0
                            except ValueError:
                                warn("Skipping unconvertible ppa value %s" % (cols[ppa_col]))
                                continue

                        if chrom in seen_chrom_pos:
                            if pos not in seen_chrom_pos[chrom]:
                                #make up a beta
                                assert(var_p_threshold is not None)
                                (p, beta, se, freq) = (var_p_threshold, 1, None, None)
                                chrom_pos_p_beta_se_freq[chrom].append((pos, p, beta, se, freq))
                                seen_chrom_pos[chrom].add(pos)
                                if chrom not in added_chrom_pos:
                                    added_chrom_pos[chrom] = set()
                                added_chrom_pos[chrom].add(pos)

                            if chrom not in input_credible_set_info:
                                input_credible_set_info[chrom] = {}
                            if cs_id not in input_credible_set_info[chrom]:
                                input_credible_set_info[chrom][cs_id] = []
                            input_credible_set_info[chrom][cs_id].append((pos, ppa))

            if total_num_vars == 0:
                bail("Didn't read in any variants!")

            gene_output_data = {}
            total_prob_causal = 0

            #run through twice
            #first, learn the window function
            closest_dist_Y = np.array([])
            closest_dist_X = np.array([])
            window_fun_intercept = None
            window_fun_slope = None

            var_all_p = np.array([])

            #store the gene probabilities for each signal
            gene_bf_data = []
            gene_bf_data_detect = []
            gene_prob_rows = []
            gene_prob_rows_detect = []
            gene_prob_cols = []
            gene_prob_cols_detect = []
            gene_prob_genes = []
            gene_prob_col_num = 0
            gene_covariate_genes = []
            self.huge_signals = []
            self.huge_signal_posteriors = []
            self.huge_signal_posteriors_for_regression = []
            self.huge_signal_sum_gene_cond_probabilities = []
            self.huge_signal_sum_gene_cond_probabilities_for_regression = []
            self.huge_signal_mean_gene_pos = []
            self.huge_signal_mean_gene_pos_for_regression = []
            self.gene_covariates = None
            self.gene_covariates_mask = None
            self.gene_covariate_names = None
            self.gene_covariate_directions = None
            self.gene_covariate_intercept_index = None
            self.gene_covariate_adjustments = None

            #second, compute the huge scores
            for learn_params in [True, False]:
                index_var_chrom_pos_ps = {}
                if learn_params:
                    log("Learning window function and allelic var scale factor")
                else:
                    log("Calculating GWAS HuGE scores")

                for chrom in chrom_pos_p_beta_se_freq:

                    #log("Processing chrom %s" % chrom, TRACE)
                    #convert all of these to np arrays sorted by chromosome
                    #sorted arrays of variant positions and p-values

                    chrom_pos_p_beta_se_freq[chrom].sort(key=lambda k: k[0])
                    vars_zipped = list(zip(*chrom_pos_p_beta_se_freq[chrom]))

                    if len(vars_zipped) == 0:
                        continue

                    var_pos = np.array(vars_zipped[0], dtype=float)
                    var_p = np.array(vars_zipped[1], dtype=float)
                    var_beta = np.array(vars_zipped[2], dtype=float)
                    var_se = np.array(vars_zipped[3], dtype=float)

                    (var_p, var_beta, var_se) = self._complete_p_beta_se(var_p, var_beta, var_se)

                    var_z = var_beta / var_se
                    var_se2 = np.square(var_se)

                    #this will vary slightly by chromosome but probably okay
                    mean_n = np.mean(1 / var_se2)

                    #sorted arrays of gene positions and p-values
                    if chrom not in gene_chrom_name_pos:
                        warn("Could not find chromosome %s in --gene-loc-file; skipping for now" % chrom)
                        continue

                    index_var_chrom_pos_ps[chrom] = []

                    gene_chrom_name_pos[chrom].sort(key=lambda k: k[1])
                    gene_zipped = list(zip(*gene_chrom_name_pos[chrom]))

                    #gene_names is array of the unique gene names
                    #gene_index_to_name_index is an array of the positions (each gene has multiple) and tells us which gene name corresponds to each position
                    gene_names_non_unique = np.array(gene_zipped[0])

                    gene_names, gene_index_to_name_index = np.unique(gene_names_non_unique, return_inverse=True)
                    gene_name_to_index = self._construct_map_to_ind(gene_names)
                    gene_pos = np.array(gene_zipped[1])

                    #get a map from position to gene
                    pos_to_gene_prob = None
                    if chrom_pos_to_gene_prob is not None and chrom in chrom_pos_to_gene_prob:
                        pos_to_gene_prob = chrom_pos_to_gene_prob[chrom]                        

                    #gene_prob_causal = np.full(len(gene_names), self.background_prior)

                    exon_interval_tree = None
                    interval_to_gene = None
                    if exons_loc_file is not None and chrom in chrom_to_interval_tree:
                        exon_interval_tree = chrom_to_interval_tree[chrom]
                        interval_to_gene = chrom_interval_to_gene[chrom]

                    def __get_closest_gene_indices(region_pos):
                        gene_indices = np.searchsorted(gene_pos, region_pos)
                        gene_indices[gene_indices == len(gene_pos)] -= 1

                        #look to the left and the right to see which gene closer
                        lower_mask = np.abs(region_pos - gene_pos[gene_indices - 1]) < np.abs(region_pos - gene_pos[gene_indices])
                        gene_indices[lower_mask] = gene_indices[lower_mask] - 1
                        return gene_indices

                    def __get_gene_posterior(region_pos, full_prob, window_fun_slope, window_fun_intercept, exon_interval_tree=None, interval_to_gene=None, pos_to_gene_prob=None, max_offset=20, cap=True, do_print=True):

                        #TODO: read in file of coding variants and set those to 95% for the closest gene, rather than using the gaussian below
                        closest_gene_indices = __get_closest_gene_indices(region_pos)

                        var_offset_prob = np.zeros((max_offset * 2 + 1, len(region_pos)))
                        var_gene_index = np.full((max_offset * 2 + 1, len(region_pos)), -1)

                        offsets = np.arange(-max_offset,max_offset+1)
                        var_offset_prob = np.zeros((len(offsets), len(region_pos)))
                        var_gene_index = np.full(var_offset_prob.shape, -1)
                        cur_gene_indices = np.add.outer(offsets, closest_gene_indices)
                        cur_gene_indices[cur_gene_indices >= len(gene_pos)] = len(gene_pos) - 1
                        cur_gene_indices[cur_gene_indices <= 0] = 0

                        prob_causal_odds = np.exp(window_fun_slope * np.abs(gene_pos[cur_gene_indices] - region_pos) + window_fun_intercept)

                        cur_prob_causal = full_prob * (prob_causal_odds / (1 + prob_causal_odds))
                        cur_prob_causal[cur_prob_causal < 0] = 0

                        #take only the maximum value across all genes, since each gene can have multiple indices, 
                        #the following code generates a mask of all of the spots that contain the maximum value per group
                        #to do so though it has to sort the arrays
                        groups = gene_index_to_name_index[cur_gene_indices]
                        data = copy.copy(cur_prob_causal)
                        order = np.lexsort((data, groups), axis=0)

                        order2 = np.arange(groups.shape[1])
                        groups2 = groups[order, order2]
                        data2 = data[order, order2]
                        max_by_group_mask = np.empty(groups2.shape, 'bool')
                        max_by_group_mask[-1,:] = True
                        max_by_group_mask[:-1,:] = groups2[1:,:] != groups2[:-1,:]

                        #now "unsort" the mask
                        rev_order = np.empty_like(order)
                        rev_order[order, order2] = np.repeat(np.arange(order.shape[0]), order.shape[1]).reshape(order.shape[0], order.shape[1])
                        rev_max_by_group_mask = max_by_group_mask[rev_order, order2]

                        #need to keep only the maximum probability for each gene for each variant (in case some genes appear multiple times)
                        #zero out the values that are not max by group
                        cur_prob_causal[~rev_max_by_group_mask] = 0

                        var_offset_prob = cur_prob_causal
                        var_gene_index = gene_index_to_name_index[cur_gene_indices]


                        def __add_var_rows(_var_inds, _gene_prob_lists, _var_offset_prob, _var_gene_index):
                            #var_inds: indices into var_gene_index and var_offset_probs
                            #_gene_prob: list of list of (gene, prob) pairs; outer list same length as var_inds
                            var_to_seen_genes = {}
                            num_added = 0
                            for i in range(len(_var_inds)):
                                cur_var_index = _var_inds[i]
                                if cur_var_index not in var_to_seen_genes:
                                    var_to_seen_genes[cur_var_index] = set()
                                for cur_gene,cur_prob in _gene_prob_lists[i]:
                                    if cur_gene in gene_name_to_index:
                                        cur_gene_index = gene_name_to_index[cur_gene]
                                        if cur_gene_index not in var_to_seen_genes[cur_var_index]:
                                            var_to_seen_genes[cur_var_index].add(cur_gene_index)
                                            if num_added < len(var_to_seen_genes[cur_var_index]):
                                                _var_offset_prob = np.vstack((_var_offset_prob, np.zeros((1, _var_offset_prob.shape[1]))))
                                                _var_gene_index = np.vstack((_var_gene_index, np.zeros((1, _var_gene_index.shape[1]))))
                                                num_added += 1

                                            #should we really set it to be zero? I think this would render the next line of code (multiplying by 1 - cur_prob) to do nothing

                                            #and I think that next line of code is correct
                                            #first need to set anything else with this index to be 0
                                            #_var_offset_prob[_var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] = 0

                                            #then scale everything non-zero down to account for likelihood that the variant is actually coding
                                            _var_offset_prob[_var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] *= (1 - cur_prob)

                                            #this is where to write exon probability
                                            row_index = _var_offset_prob.shape[0] - (num_added - len(var_to_seen_genes[cur_var_index])) - 1
                                            _var_offset_prob[row_index,cur_var_index] = full_prob[cur_var_index] * cur_prob
                                            _var_gene_index[row_index,cur_var_index] = cur_gene_index

                            return((_var_offset_prob, _var_gene_index))



                        if exon_interval_tree is not None and interval_to_gene is not None:
                            #now add in a row for the exons
                            #this is the list of region_pos that overlap an exon
                            (region_with_overlap_inds, overlapping_interval_starts, overlapping_interval_stops) = exon_interval_tree.find(region_pos, region_pos)
                            coding_var_linkage_prob = np.maximum(np.exp(window_fun_slope + window_fun_intercept)/(1+np.exp(window_fun_slope + window_fun_intercept)), 0.95)

                            #if True:
                            #this needs to have the gene, prob corresponding to each position
                            gene_lists = [interval_to_gene[(overlapping_interval_starts[i], overlapping_interval_stops[i])] for i in range(len(region_with_overlap_inds))]
                            gene_prob_lists = []
                            for i in range(len(gene_lists)):
                                gene_prob_lists.append(list(zip(gene_lists[i], [coding_var_linkage_prob for j in range(len(gene_lists[i]))])))

                            var_offset_prob, var_gene_index = __add_var_rows(region_with_overlap_inds, gene_prob_lists, var_offset_prob, var_gene_index)
#                            else:
#                                #TODO: DELETE THIS
#                                #THIS IS OLD CODE IN CASE ABOVE DOESN'T WORK
#
#                                #append a column to the var_offset_prob and var_gene_index corresponding to the exon
#                                #may need to append more than one column if a variant is in exons of more than one gene
#                                var_to_seen_genes = {}
#                                num_added = 0
#                                for i in range(len(region_with_overlap_inds)):
#                                    cur_var_index = region_with_overlap_inds[i]
#                                    if cur_var_index not in var_to_seen_genes:
#                                        var_to_seen_genes[cur_var_index] = set()
#                                    cur_genes = interval_to_gene[(overlapping_interval_starts[i], overlapping_interval_stops[i])]
#                                    for cur_gene in cur_genes:
#                                        if cur_gene in gene_name_to_index:
#
#                                            cur_gene_index = gene_name_to_index[cur_gene]
#                                            if cur_gene_index not in var_to_seen_genes[cur_var_index]:
#                                                var_to_seen_genes[cur_var_index].add(cur_gene_index)
#                                                if num_added < len(var_to_seen_genes[cur_var_index]):
#                                                    var_offset_prob = np.vstack((var_offset_prob, np.zeros((1, var_offset_prob.shape[1]))))
#                                                    var_gene_index = np.vstack((var_gene_index, np.zeros((1, var_gene_index.shape[1]))))
#                                                    num_added += 1
#                                                #first need to set anything else with this index to be 0
#                                                var_offset_prob[var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] = 0
#                                                #then scale everything non-zero down to account for likelihood that the variant is actually coding
#                                                var_offset_prob[var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] *= (1 - coding_var_linkage_prob)
#                                                #this is where to write exon probability
#                                                row_index = var_offset_prob.shape[0] - (num_added - len(var_to_seen_genes[cur_var_index])) - 1
#                                                var_offset_prob[row_index,cur_var_index] = full_prob[cur_var_index] * coding_var_linkage_prob
#                                                var_gene_index[row_index,cur_var_index] = cur_gene_index


                        if pos_to_gene_prob is not None:
                            gene_prob_lists = []
                            for i in range(len(region_pos)):
                                probs = []
                                if region_pos[i] in pos_to_gene_prob:
                                    probs = pos_to_gene_prob[region_pos[i]]
                                gene_prob_lists.append(probs)
                            var_offset_prob, var_gene_index = __add_var_rows(range(len(region_pos)), gene_prob_lists, var_offset_prob, var_gene_index)

                        var_gene_index = var_gene_index.astype(int)

                        #first normalize not accounting for any dependencies across genes

                        #scale_raw_closest_gene: set everything to have the closest gene as closest gene prob
                        #cap_raw_closest_gene: set everything to have probability no greater than closest gene prob

                        if scale_raw_closest_gene or cap_raw_closest_gene:
                            var_offset_prob_max = var_offset_prob.max(axis=0)
                            var_offset_norm = np.ones(full_prob.shape)
                            var_offset_norm[var_offset_prob_max != 0] = full_prob[var_offset_prob_max != 0] * closest_gene_prob / var_offset_prob_max[var_offset_prob_max != 0]

                            if cap_raw_closest_gene:
                                cap_mask = var_offset_norm > 1
                                var_offset_norm[cap_mask] = 1
                        else:
                            var_offset_norm = 1

                        var_offset_prob *= var_offset_norm

                        def ___aggregate_var_gene_index(cur_var_offset_prob):

                            cur_gene_indices, idx = np.unique(var_gene_index.ravel(), return_inverse=True)
                            cur_gene_prob_causal = np.bincount(idx, weights=cur_var_offset_prob.ravel())

                            #remove the very low ones
                            non_zero_mask = cur_gene_prob_causal > 0.001 * np.max(cur_gene_prob_causal)

                            cur_gene_prob_causal = cur_gene_prob_causal[non_zero_mask]
                            cur_gene_indices = cur_gene_indices[non_zero_mask]

                            #cap very high ones

                            cur_gene_po = None
                            if cap:
                                cur_gene_prob_causal[cur_gene_prob_causal > 0.999] = 0.999
                                cur_gene_po = cur_gene_prob_causal / (1 - cur_gene_prob_causal)

                            return (cur_gene_prob_causal, cur_gene_indices, cur_gene_po)

                        (cur_gene_prob_causal_no_norm, cur_gene_indices_no_norm, cur_gene_po_no_norm) = ___aggregate_var_gene_index(var_offset_prob)

                        #now do it normalized
                        var_offset_prob_sum = np.sum(var_offset_prob, axis=0)
                        var_offset_prob_sum[var_offset_prob_sum < 1] = 1
                        var_offset_prob_norm = var_offset_prob / var_offset_prob_sum
                        (cur_gene_prob_causal_norm, cur_gene_indices_norm, cur_gene_po_norm) = ___aggregate_var_gene_index(var_offset_prob_norm)

                        return (cur_gene_prob_causal_no_norm, cur_gene_indices_no_norm, cur_gene_po_no_norm, cur_gene_prob_causal_norm, cur_gene_indices_norm)

                    if learn_params:

                        #randomly sample 100K variants
                        region_vars = np.full(len(var_pos), False)
                        number_needed = 100000

                        region_vars[np.random.random(len(region_vars)) < (float(number_needed) / total_num_vars)] = True

                        closest_gene_indices = __get_closest_gene_indices(var_pos[region_vars])

                        closest_dists = np.abs(gene_pos[closest_gene_indices] - var_pos[region_vars])
                        closest_dists = closest_dists[closest_dists <= max_closest_gene_dist]

                        closest_dist_X = np.append(closest_dist_X, closest_dists)
                        closest_dist_Y = np.append(closest_dist_Y, np.full(len(closest_dists), closest_gene_prob))

                        var_all_p = np.append(var_all_p, var_p)

                        max_offset = 200

                        #new, vectorized
                        offsets = np.arange(-max_offset,max_offset+1)
                        cur_gene_indices = np.add.outer(offsets, closest_gene_indices)
                        cur_gene_indices[cur_gene_indices >= len(gene_pos)] = len(gene_pos) - 1
                        cur_gene_indices[cur_gene_indices <= 0] = 0

                        #ignore any that are actually the closest gene
                        cur_mask = (gene_names[gene_index_to_name_index[cur_gene_indices]] != gene_names[gene_index_to_name_index[closest_gene_indices]])
                        #remove everything above the maximum
                        non_closest_dists = np.abs(gene_pos[cur_gene_indices] - var_pos[region_vars])

                        cur_mask = np.logical_and(cur_mask, non_closest_dists <= max_closest_gene_dist)

                        #maximum is used here to avoid divide by 0
                        #any zeros will get subsetted out by cur_mask anyway
                        non_closest_probs = (np.full(non_closest_dists.shape, (1 - closest_gene_prob) / np.maximum(np.sum(cur_mask, axis=0), 1)))[cur_mask]
                        non_closest_dists = non_closest_dists[cur_mask]

                        closest_dist_X = np.append(closest_dist_X, non_closest_dists)

                        closest_dist_Y = np.append(closest_dist_Y, np.full(len(non_closest_dists), non_closest_probs))

                    else:

                        if correct_huge:

                            #first 
                            max_gene_offset = 500

                            gene_offsets = np.arange(max_gene_offset+1)

                            gene_start_indices = np.zeros(len(gene_names), dtype=int)
                            gene_end_indices = np.zeros(len(gene_names), dtype=int)
                            gene_num_indices = np.zeros(len(gene_names), dtype=int)

                            gene_name_to_ind = self._construct_map_to_ind(gene_names)
                            for i in range(len(gene_names_non_unique)):
                                gene_name_ind = gene_name_to_ind[gene_names_non_unique[i]]
                                if gene_start_indices[gene_name_ind] == 0:
                                    gene_start_indices[gene_name_ind] = i
                                gene_end_indices[gene_name_ind] = i
                                gene_num_indices[gene_name_ind] += 1

                            #these store the indices of genes to the left and right
                            genes_higher_indices = np.add.outer(gene_offsets, gene_end_indices).astype(int)
                            genes_ignore_indices = np.full(genes_higher_indices.shape, False)
                            genes_ignore_indices[genes_higher_indices >= len(gene_pos)] = True
                            genes_higher_indices[genes_higher_indices >= len(gene_pos)] = len(gene_pos) - 1
                            genes_lower_indices = np.add.outer(-gene_offsets, gene_start_indices).astype(int)
                            genes_ignore_indices[genes_lower_indices <= 0] = True
                            genes_lower_indices[genes_lower_indices <= 0] = 0

                            #ignore any that are actually the gene itself

                            higher_ignore_mask = np.logical_or(genes_ignore_indices, (gene_names[gene_index_to_name_index[genes_higher_indices]] == gene_names[gene_index_to_name_index[gene_end_indices]]))
                            lower_ignore_mask = np.logical_or(genes_ignore_indices, (gene_names[gene_index_to_name_index[genes_lower_indices]] == gene_names[gene_index_to_name_index[gene_start_indices]]))

                            right_dists = (gene_pos[genes_higher_indices] - gene_pos[gene_end_indices]).astype(float)

                            right_dists[higher_ignore_mask] = np.inf
                            right_dists[right_dists == 0] = 1

                            left_dists = (gene_pos[gene_start_indices] - gene_pos[genes_lower_indices]).astype(float)
                            left_dists[lower_ignore_mask] = np.inf
                            left_dists[left_dists == 0] = 1

                            # distance to next closest gene (left and right)

                            right_dist = np.min(right_dists, axis=0)
                            left_dist = np.min(left_dists, axis=0)

                            # sum of 1/distance (or logit distance) to 5 or 10 nearest genes (left and right)

                            right_sum = np.sum(1.0 / right_dists, axis=0)
                            left_sum = np.sum(1.0 / left_dists, axis=0)
                            right_left_sum = right_sum + left_sum

                            # number of genes within 1 Mb or 10 Mb (left and right)
                            large_dist = 250000
                            small_dist = 50000

                            num_right_small = np.sum(right_dists < small_dist, axis=0)
                            num_left_small = np.sum(left_dists < small_dist, axis=0)

                            num_right_large = np.sum(right_dists < large_dist, axis=0)
                            num_left_large = np.sum(left_dists < large_dist, axis=0)

                            num_small = num_right_small + num_left_small
                            num_large = num_right_large + num_left_large

                            # expanse of the gene
                            gene_size = gene_pos[gene_end_indices] - gene_pos[gene_start_indices]

                            # number of locations
                            #gene_num_indices

                            #sum of linkqge probabilities
                            chrom_start = np.max((np.min(gene_pos) - 1e6, 0))
                            chrom_end = np.max(gene_pos) + 1e6
                            #space them evenly, with spacing equal to average distance between SNPs in a 10e6 SNP GWAS
                            sim_variant_positions = np.linspace(chrom_start, chrom_end, int((chrom_end - chrom_start) / (3e9/2e5)), dtype=int)

                            (sim_gene_prob_causal_orig, sim_gene_indices, sim_gene_po, sim_gene_prob_causal_norm_orig, sim_gene_indices_norm) = __get_gene_posterior(sim_variant_positions, np.ones(len(sim_variant_positions)), window_fun_slope, window_fun_intercept, max_offset=20, cap=False, do_print=False)

                            #have to map these over to the original indices in case the sim_gene_prob_causal_orig was missing some genes
                            sim_gene_prob_causal = np.zeros(len(gene_names))
                            for i in range(len(sim_gene_indices)):
                                sim_gene_prob_causal[sim_gene_indices[i]] = sim_gene_prob_causal_orig[i]
                            sim_gene_prob_causal_norm = np.zeros(len(gene_names))
                            for i in range(len(sim_gene_indices_norm)):
                                sim_gene_prob_causal_norm[sim_gene_indices_norm[i]] = sim_gene_prob_causal_norm_orig[i]

                            cur_gene_covariates = np.vstack((right_left_sum, num_right_large, num_left_large, gene_num_indices, sim_gene_prob_causal, np.ones(len(gene_names)))).T

                            #OLD ONES
                            #cur_gene_covariates = np.vstack((sim_gene_prob_causal, np.ones(len(gene_names)))).T
                            #cur_gene_covariates = np.vstack((right_left_sum, num_right_large, num_left_large, num_right_small, num_left_small, gene_num_indices, sim_gene_prob_causal, np.ones(len(gene_names)))).T
                            #cur_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_right_large, num_left_large, gene_num_indices, sim_gene_prob_causal, sim_gene_prob_causal_norm, np.ones(len(gene_names)))).T
                            #cur_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_right_large, num_left_large, sim_gene_prob_causal, sim_gene_prob_causal_norm, np.ones(len(gene_names)))).T
                            #cur_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_right_large, num_left_large, gene_size, gene_num_indices, sim_gene_prob_causal, sim_gene_prob_causal_norm, np.ones(len(gene_names)))).T
                            #cur_gene_covariates = np.vstack((sim_gene_prob_causal, np.ones(len(gene_names)))).T
                            #cur_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_small, num_large, gene_size, gene_num_indices, np.ones(len(gene_names)))).T
                            #cur_gene_covariates = np.vstack((np.maximum(right_dist, left_dist), right_left_sum, np.minimum(num_right_small, num_left_small), np.minimum(num_right_large, num_left_large), gene_size, gene_num_indices, np.ones(len(gene_names)))).T

                            if self.gene_covariates is None:

                                self.gene_covariates = cur_gene_covariates

                                self.gene_covariate_names = ["right_left_sum_inv", "num_right_%s" % large_dist, "num_left_%s" % large_dist, "gene_num_indices", "sim_prob_causal", "intercept"]
                                self.gene_covariate_directions = np.array([-1, -1, -1, 1, 1, 0])

                                self.gene_covariate_slope_defaults = np.array([-0.02321564, -0.00182764, -0.00315613,  0.00824289,  0.00316042, 0.08495138])
                                self.total_qc_metric_betas_defaults = [-0.01659398, -0.03525455, -0.04813412,  0.00553828, -0.39453483, -0.53903559]
                                self.total_qc_metric_intercept_defaults = 0.98859127
                                self.total_qc_metric2_betas_defaults = [-0.00092923, -0.25170301, -0.25994094,  0.13700834, -0.10948609, -0.510157  ]
                                self.total_qc_metric2_intercept_defaults = 1.70380708

                                #OLD ONES
                                #self.gene_covariate_names = ["sim_prob_causal", "intercept"]
                                #self.gene_covariate_names = ["right_left_sum_inv", "num_right_%s" % large_dist, "num_left_%s" % large_dist, "num_right_%s" % small_dist, "num_left_%s" % small_dist, "gene_num_indices", "sim_prob_causal", "intercept"]
                                #self.gene_covariate_names = ["right_dist", "left_dist", "right_left_sum_inv", "num_right_%s" % large_dist, "num_left_%s" % large_dist, "sim_prob_causal", "sim_prob_causal_norm", "intercept"]
                                #self.gene_covariate_names = ["right_dist", "left_dist", "right_left_sum_inv", "num_right_%s" % large_dist, "num_left_%s" % large_dist, "gene_size", "gene_num_indices", "sim_prob_causal", "sim_prob_causal_norm", "intercept"]
                                #self.gene_covariate_names = ["sim_prob_causal", "intercept"]
                                #self.gene_covariate_names = ["max_dist", "right_left_sum_inv", "min_num_%s" % small_dist, "min_num_%s" % large_dist, "gene_size", "gene_num_indices", "intercept"]

                                self.gene_covariate_intercept_index = len(self.gene_covariate_names) - 1

                            else:
                                self.gene_covariates = np.vstack((self.gene_covariates, cur_gene_covariates))

                            gene_covariate_genes += list(gene_names)

                    #now onto variants


                    #Z-score based one:
                    #K=-0.439
                    #np.sqrt(1 + K) * np.exp(-np.square(var_z) / 2 * (K) / (1 + K))
                    #or, for which sample size doesn't matter:
                    #K=-0.439 / np.mean(var_n)
                    #np.sqrt(1 + var_n * K) * np.exp(-np.square(var_z) / 2 * (var_n * K) / (1 + var_n * K))

                    #var_log_bf = np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k)
                    var_log_bf = -np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k)

                    if separate_detect:
                        var_log_bf_detect = -np.log(np.sqrt(1 + allelic_var_k_detect)) + 0.5 * np.square(var_z) * allelic_var_k_detect / (1 + allelic_var_k_detect)
                    else:
                        var_log_bf_detect = copy.copy(var_log_bf)

                    #now calculate the posteriors
                    var_posterior = var_log_bf + np.log(gwas_prior_odds)
                    if separate_detect:
                        var_posterior_detect = var_log_bf_detect + np.log(gwas_prior_odds_detect)
                    else:
                        var_posterior_detect = copy.copy(var_posterior)

                    max_log = 15



                    if separate_detect:
                        update_posterior = [var_posterior, var_posterior_detect]
                    else:
                        update_posterior = [var_posterior]

                    for cur_var_posterior in update_posterior:
                        max_mask = cur_var_posterior < max_log
                        cur_var_posterior[~max_mask] = 1
                        cur_var_posterior[max_mask] = np.exp(cur_var_posterior[max_mask])
                        cur_var_posterior[max_mask] = cur_var_posterior[max_mask] / (1 + cur_var_posterior[max_mask])

                    if not separate_detect:
                        var_posterior_detect = copy.copy(var_posterior)

                    variants_keep = np.full(len(var_pos), True)
                    qc_fail = 1 / var_se2 < min_n_ratio * mean_n
                    variants_keep[qc_fail] = False

                    #make sure to add in additional credible set ids
                    if not learn_params and chrom in added_chrom_pos:
                        for cur_pos in added_chrom_pos[chrom]:
                            variants_keep[var_pos == cur_pos] = True

                    #filter down for efficiency
                    var_pos = var_pos[variants_keep]
                    var_p = var_p[variants_keep]
                    var_beta = var_beta[variants_keep]
                    var_se = var_se[variants_keep]
                    var_se2 = var_se2[variants_keep]
                    var_log_bf = var_log_bf[variants_keep]
                    var_log_bf_detect = var_log_bf_detect[variants_keep]
                    var_posterior = var_posterior[variants_keep]
                    var_posterior_detect = var_posterior_detect[variants_keep]

                    var_logp = -np.log(var_p) / np.log(10)

                    var_freq = None
                    if freq_col is not None:
                        var_freq = np.array(vars_zipped[4], dtype=float)[variants_keep]
                        var_freq[var_freq > 0.5] = 1 - var_freq[var_freq > 0.5]

                    variants_left = np.full(len(var_pos), True)
                    cs_ignore = np.full(len(var_pos), False)
                    while np.sum(variants_left) > 0:

                        cond_prob = None
                        cond_prob_detect = None
                        is_input_cs = False
                        if not learn_params and chrom in input_credible_set_info and len(input_credible_set_info[chrom].keys()) > 0:

                            cur_cs_id = list(input_credible_set_info[chrom].keys())[0]
                            cur_cs_vars = input_credible_set_info[chrom][cur_cs_id]
                            is_input_cs = True

                            region_vars = np.full(len(var_pos), False)

                            cond_prob = np.zeros(len(var_pos))
                            for pos_ppa in cur_cs_vars:
                                pos = pos_ppa[0]
                                ppa = pos_ppa[1]
                                mask = np.logical_and(variants_left, var_pos == pos)

                                if np.sum(mask) > 0:
                                    region_vars[mask] = True
                                    if ppa is not None:
                                        cond_prob[mask] = ppa

                            #all of the credible set variants have been used
                            if np.sum(region_vars) == 0:
                                del input_credible_set_info[chrom][cur_cs_id]
                                continue

                            cur_cs_ignore = np.logical_and(var_pos > np.min(var_pos[region_vars]) - credible_set_span, var_pos < np.max(var_pos[region_vars]) + credible_set_span)

                            if np.sum(cond_prob) > 0:
                                cond_prob /= np.sum(cond_prob)

                                i = np.argmax(cond_prob)
                                cond_prob = cond_prob[region_vars]
                                cond_prob_detect = copy.copy(cond_prob)
                            else:
                                cond_prob = None
                                if np.sum(mask) > 0:
                                    i = np.where(mask)[0][0]
                                else:
                                    #find the highest variant within cur_cs_ignore
                                    cs_variant_window = np.logical_and(np.logical_and(variants_left, ~cs_ignore), cur_cs_ignore)
                                    if np.sum(cs_variant_window) > 0:
                                        #if there is at least one, then we use the minimum p-value

                                        #get the lowest p-value remaining variant
                                        cs_variant_inds = np.where(cs_variant_window)[0]
                                        i = cs_variant_inds[np.argmin(var_p[cs_variant_inds])]
                                    else:
                                        #otherwise reset
                                        is_input_cs = False

                            if is_input_cs:
                                cs_ignore = np.logical_or(cs_ignore, cur_cs_ignore)
                                del input_credible_set_info[chrom][cur_cs_id]

                        #if we didn't have credible set, or it didn't have PPA, then we go through here
                        if cond_prob is None:

                            #if it wasn't a credible set, we select the lead SNP. Otherwise we selected above
                            if not is_input_cs:

                                if not learn_params:
                                    variants_left = np.logical_and(variants_left, ~cs_ignore)

                                #get the lowest p-value remaining variant
                                variants_left_inds = np.where(variants_left)[0]
                                i = variants_left_inds[np.argmin(var_p[variants_left_inds])]

                                #get all variants in the region
                                #TODO: here is where we would ideally clump either by
                                #1. reading in LD (if we have it)

                                #find variants within 100kb, extending as needed if we have some above 
                                #log("Processing variant %s:%d" % (chrom, var_pos[i]), TRACE)

                            #we will do this if there was no credible set, or if the credible set just gave us the top variant
                            #if it just gave us the top variant, then we expand around the lead SNP in the credible set

                            region_vars = np.logical_and(var_pos >= var_pos[i] - signal_window_size, var_pos <= var_pos[i] + signal_window_size)

                            region_inds = np.where(region_vars)[0]
                            assert(len(region_inds) > 0)

                            #extend the region until the distance to the last significant snp is greater than the signal_min_sep

                            increase_ratio = 1.3
                            self._record_param("p_value_increase_ratio_for_sep_signal", increase_ratio)

                            region_ind = region_inds[0] - 1
                            last_significant_snp = region_inds[0]
                            while region_ind > 0 and np.abs(var_pos[region_ind] - var_pos[last_significant_snp]) < signal_min_sep:
                                if var_p[region_ind] < max_signal_p:

                                    if var_p[region_ind] < var_p[last_significant_snp]:
                                        #check if it starts to increase after it
                                        cur_block = np.logical_and(np.logical_and(var_pos >= var_pos[region_ind], var_pos < var_pos[region_ind] + signal_min_sep), var_p < max_signal_p)

                                        prev_block = np.logical_and(np.logical_and(var_pos >= var_pos[region_ind] + signal_min_sep, var_pos < var_pos[region_ind] + 2 * signal_min_sep), var_p < max_signal_p)

                                        #if the mean p-value of significant SNPs decreases relative to the previous one, stop extending
                                        if np.sum(prev_block) == 0 or np.sum(cur_block) == 0 or np.mean(var_logp[cur_block]) > increase_ratio * np.mean(var_logp[prev_block]):
                                            break

                                    last_significant_snp = region_ind

                                    region_vars[region_ind:region_inds[0]] = True

                                region_ind -= 1

                            region_ind = region_inds[-1] + 1
                            last_significant_snp = region_inds[0]
                            while region_ind < len(var_pos) and np.abs(var_pos[region_ind] - var_pos[last_significant_snp]) < signal_min_sep:
                                if var_p[region_ind] < max_signal_p:
                                    #if this is increasing, check to see if we can break
                                    if var_p[region_ind] < var_p[last_significant_snp]:
                                        cur_block = np.logical_and(np.logical_and(var_pos <= var_pos[region_ind], var_pos > var_pos[region_ind] - signal_min_sep), var_p < max_signal_p)
                                        prev_block = np.logical_and(np.logical_and(var_pos <= var_pos[region_ind] - signal_min_sep, var_pos > var_pos[region_ind] - 2 * signal_min_sep), var_p < max_signal_p)

                                        #if the mean p-value of significant SNPs decreases relative to the previous one, stop extending
                                        if np.sum(prev_block) == 0 or np.sum(cur_block) == 0 or np.mean(var_logp[cur_block]) > increase_ratio * np.mean(var_logp[prev_block]):
                                            break

                                    last_significant_snp = region_ind

                                    region_vars[region_inds[-1]:region_ind] = True
                                region_ind += 1

                            #if we have MAF, approximate LD by MAF
                            if var_freq is not None:

                                #maximum LD occurs when genotype carriers completely overlap (one is subset of another)
                                #(E[XY] - E[X]E[Y]) / sqrt((E[X^2] - E[X]^2)(E[Y^2] - E[Y]^2))
                                #(MAF_MIN - MAF_MAX * MAF_MIN) / sqrt(MAF_MAX * (1 - MAF_MAX) * MAF_MIN * (1 - MAF_MIN)
                                #MAF_MIN * (1 - MAF_MAX) / sqrt(MAF_MAX * (1 - MAF_MAX) * MAF_MIN * (1 - MAF_MIN)
                                #sqrt(MAF_MIN * (1 - MAF_MAX)) / sqrt((1 - MAF_MAX) * MAF_MIN)

                                max_ld = np.sqrt((var_freq[i] * (1 - var_freq)) / (var_freq * (1 - var_freq[i])))
                                max_ld[var_freq[i] > var_freq] = 1.0 / max_ld[var_freq[i] > var_freq]

                                int_mask = np.logical_and(region_vars, max_ld < max_clump_ld)
                                if np.sum(int_mask) > 0:
                                    argminp = np.argmin(var_p[int_mask])

                                #for variants with frequencies that imply they cannot have LD above max_clump_ld with index var,
                                #remove them from this clump
                                region_vars[max_ld < max_clump_ld] = False

                            if signal_max_logp_ratio is not None:
                                region_vars[var_logp/var_logp[i] < signal_max_logp_ratio] = False

                        #now remove all of the variants that have been seen
                        left_mask = variants_left[region_vars]
                        region_vars = np.logical_and(region_vars, variants_left)
                        #set these to not be seen again
                        variants_left[region_vars] = False

                        index_var_chrom_pos_ps[chrom].append((var_pos[i], var_p[i]))


                        #let's always treat the top posterior as the signal
                        #this will always hold unless we chose a credible set with a variant that doesn't have a p-value

                        sig_posterior = np.max(var_posterior[region_vars])
                        sig_posterior_detect = np.max(var_posterior_detect[region_vars])

                        min_pos = np.min(var_pos[region_vars])
                        max_pos = np.max(var_pos[region_vars])
                        #log("%d-%d (%d)" % (min_pos, max_pos, max_pos - min_pos))
                        #if not learn_params:
                            #log("Index SNP %d=%d; region=%d-%d; logp=%.3g-%.3g" % (i,var_pos[i], np.min(var_pos[region_vars]), np.max(var_pos[region_vars]), np.min(var_logp[region_vars]), np.max(var_logp[region_vars])), TRACE)
                        #log("Variant:",var_pos[i],"P:",var_p[i],"POST:",sig_posterior,"MIN_POS:",min_pos,"MAX_POS:",max_pos,"NUM:",np.sum(region_vars))
                        #m = np.where(var_pos == 84279410.0)[0]


                        if cond_prob is None:
                            #now find the conditional posteriors of all of the variants in the region
                            #use log sum exp trick

                            c = np.max(var_log_bf[region_vars])
                            c_detect = np.max(var_log_bf_detect[region_vars])

                            log_sum_bf = c + np.log(np.sum(np.exp(var_log_bf[region_vars] - c)))
                            log_sum_bf_detect = c_detect + np.log(np.sum(np.exp(var_log_bf_detect[region_vars] - c_detect)))

                            log_rel_bf = var_log_bf[region_vars] - log_sum_bf
                            log_rel_bf_detect = var_log_bf_detect[region_vars] - log_sum_bf_detect

                            cond_prob = log_rel_bf
                            cond_prob[cond_prob > max_log] = 1
                            cond_prob[cond_prob < max_log] = np.exp(cond_prob[cond_prob < max_log])

                            cond_prob_detect = log_rel_bf_detect
                            cond_prob_detect[cond_prob_detect > max_log] = 1
                            cond_prob_detect[cond_prob_detect < max_log] = np.exp(cond_prob_detect[cond_prob_detect < max_log])


                        #this is the final posterior probability of association for all variants in the region
                        full_prob = cond_prob * sig_posterior
                        full_prob_detect = cond_prob_detect * sig_posterior_detect

                        if not learn_params:

                            #calculate the posteriors

                            #find out how many indices to look to the left and right of the nearest one before assigning 0 linkage probability

                            #first get all of the gene indices within the max window of each variant on each size
                            gene_index_ranges = __get_closest_gene_indices(np.vstack((var_pos[region_vars], var_pos[region_vars] - max_closest_gene_dist, var_pos[region_vars] + max_closest_gene_dist)))

                            max_num_indices = np.maximum(np.max(gene_index_ranges[0,:] - gene_index_ranges[1,:]), np.max(gene_index_ranges[2,:] - gene_index_ranges[0,:]))

                            (cur_gene_prob_causal, cur_gene_indices, cur_gene_po, cur_gene_prob_causal_norm, cur_gene_indices_norm) = __get_gene_posterior(var_pos[region_vars], full_prob, window_fun_slope, window_fun_intercept, exon_interval_tree=exon_interval_tree, interval_to_gene=interval_to_gene, pos_to_gene_prob=pos_to_gene_prob, max_offset=max_num_indices)

                            if separate_detect:
                                (cur_gene_prob_causal_detect, cur_gene_indices_detect, cur_gene_po_detect, cur_gene_prob_causal_norm_detect, cur_gene_indices_norm_detect) = __get_gene_posterior(var_pos[region_vars], full_prob_detect, window_fun_slope, window_fun_intercept, exon_interval_tree=exon_interval_tree, interval_to_gene=interval_to_gene, pos_to_gene_prob=pos_to_gene_prob, max_offset=max_num_indices)
                            else:
                                (cur_gene_prob_causal_detect, cur_gene_indices_detect, cur_gene_po_detect, cur_gene_prob_causal_norm_detect, cur_gene_indices_norm_detect) = (copy.copy(cur_gene_prob_causal), copy.copy(cur_gene_indices), copy.copy(cur_gene_po), copy.copy(cur_gene_prob_causal_norm), copy.copy(cur_gene_indices_norm))

                            gene_prob_rows += list(len(gene_prob_genes) + cur_gene_indices)
                            gene_prob_rows_detect += list(len(gene_prob_genes) + cur_gene_indices_detect)

                            gene_prob_cols += ([gene_prob_col_num] * len(cur_gene_indices))
                            gene_prob_cols_detect += ([gene_prob_col_num] * len(cur_gene_indices_detect))

                            gene_bf_data += list(cur_gene_po  / self.background_bf)
                            gene_bf_data_detect += list(cur_gene_po_detect  / self.background_bf)


                            #the total posterior (for use in scale)


                            self.huge_signals.append((chrom, var_pos[i], var_p[i], is_input_cs))

                            self.huge_signal_posteriors.append(sig_posterior)
                            self.huge_signal_posteriors_for_regression.append(sig_posterior_detect)

                            #store the marginal bayes factor
                            cur_gene_cond_prob_causal = cur_gene_prob_causal / sig_posterior
                            cur_gene_cond_prob_causal_detect = cur_gene_prob_causal_detect / sig_posterior_detect

                            #the sum of the conditional probabilities (after taking out sig posterior)
                            sum_cond_prob = np.sum(cur_gene_cond_prob_causal)
                            sum_cond_prob_detect = np.sum(cur_gene_cond_prob_causal_detect)

                            self.huge_signal_sum_gene_cond_probabilities.append(sum_cond_prob if sum_cond_prob < 1 else 1)
                            self.huge_signal_sum_gene_cond_probabilities_for_regression.append(sum_cond_prob_detect if sum_cond_prob_detect < 1 else 1)

                            #the mean of the conditional BFs
                            mean_cond_po = np.sum(cur_gene_cond_prob_causal / (1 - cur_gene_cond_prob_causal))
                            mean_cond_po_detect = np.sum(cur_gene_cond_prob_causal_detect / (1 - cur_gene_cond_prob_causal_detect))
                            self.huge_signal_mean_gene_pos.append(mean_cond_po)
                            self.huge_signal_mean_gene_pos_for_regression.append(mean_cond_po_detect)
                            gene_prob_col_num += 1

                            #now record them
                            #for i in range(len(gene_pos)):
                            #   gene_index = gene_index_to_name_index[i]
                            #    gene_name = gene_names[gene_index]
                            #    if gene_name not in gene_output_data:
                            #        gene_output_data[gene_name] = gene_prob_causal[gene_index]
                            #        total_prob_causal += gene_prob_causal[gene_index]
                            #    else:
                            #        #sanity check: same gene name should have same probability
                            #        assert(gene_prob_causal[gene_index] == gene_output_data[gene_name])

                            gene_prob_genes += list(gene_names)

                if learn_params:

                    #first update units if needed
                    unit_scale_factor = None
                    if gwas_units is not None:
                        unit_scale_factor = np.square(gwas_units)

                    index_var_ps = []
                    for chrom in index_var_chrom_pos_ps:
                        cur_pos = np.array(list(zip(*index_var_chrom_pos_ps[chrom]))[0])
                        cur_ps = np.array(list(zip(*index_var_chrom_pos_ps[chrom]))[1])
                        #now filter the variants
                        indep_window = 1e6
                        tree = IntervalTree([(x - indep_window, x + indep_window) for x in cur_pos])
                        start_to_index = dict([(cur_pos[i] - indep_window, i) for i in range(len(cur_pos))])
                        (ind_with_overlap_inds, overlapping_interval_starts, overlapping_interval_stops) = tree.find(cur_pos, cur_pos)
                        #ind_with_overlap_inds is the indices that had an overlap
                        #overlapping_interval_starts is start position of the overlapping interval; we can map these to indices by adding window size
                        assert(np.isclose(overlapping_interval_stops - overlapping_interval_starts - 2 * indep_window, np.zeros(len(overlapping_interval_stops))).all())

                        overlapping_inds = [start_to_index[i] for i in overlapping_interval_starts]

                        var_p = cur_ps[ind_with_overlap_inds]
                        overlap_var_p = cur_ps[overlapping_inds]
                        #this is a mask of the indices that are nearby another stronger variant
                        var_not_best_mask = overlap_var_p < var_p
                        #this is the list of indices

                        indep_mask = np.full(len(cur_pos), True)
                        indep_mask[ind_with_overlap_inds[var_not_best_mask]] = False

                        index_var_ps += list(cur_ps[indep_mask])

                    index_var_ps.sort()

                    index_var_ps = np.array(index_var_ps)
                    num_below_low_p = np.sum(index_var_ps < gwas_low_p)


                    self._record_param("num_below_initial_low_p", num_below_low_p)

                    log(" (%d variants below p=%.4g)" % (num_below_low_p, gwas_low_p))

                    if detect_high_power is not None or detect_low_power is not None:
                        target_max_num_variants = detect_high_power
                        target_min_num_variants = detect_low_power

                        old_low_p = gwas_low_p
                        high_or_low = None
                        if target_max_num_variants is not None and num_below_low_p > target_max_num_variants:
                            gwas_low_p = index_var_ps[target_max_num_variants]
                            high_or_low = "high"
                        if target_min_num_variants is not None and num_below_low_p < target_min_num_variants:
                            if len(index_var_ps) > target_min_num_variants:
                                gwas_low_p = index_var_ps[target_min_num_variants]
                            elif len(index_var_ps) > 0:
                                gwas_low_p = np.min(index_var_ps)
                            else:
                                gwas_low_p = 0.05
                            high_or_low = "low"

                        if high_or_low is not None:
                            self._record_param("gwas_low_p", gwas_low_p)

                            log("Detected %s power (%d variants below p=%.4g); adjusting --gwas-low-p to %.4g" % (high_or_low, num_below_low_p, old_low_p, gwas_low_p))
                            (allelic_var_k_detect, gwas_prior_odds_detect) = self.compute_allelic_var_and_prior(gwas_high_p, gwas_high_p_posterior, gwas_low_p, gwas_low_p_posterior)
                            separate_detect = True

                            if detect_adjust_huge:
                                #we have to adjust both for regression and the values used for huge scores
                                (allelic_var_k, gwas_prior_odds) = (allelic_var_k_detect, gwas_prior_odds_detect)
                                log("Using k=%.3g, po=%.3g for regression and huge scores" % (allelic_var_k_detect, gwas_prior_odds_detect))
                                self._record_params({"gwas_allelic_var_k": allelic_var_k, "gwas_prior_odds": gwas_prior_odds})
                            else:
                                log("Using k=%.3g, po=%.3g for regression only" % (allelic_var_k_detect, gwas_prior_odds_detect))
                                self._record_params({"gwas_allelic_var_k_detect": allelic_var_k_detect, "gwas_prior_odds_detect": gwas_prior_odds_detect})

                    log("Using k=%.3g, po=%.3g" % (allelic_var_k, gwas_prior_odds))
                    self._record_params({"gwas_allelic_var_k": allelic_var_k, "gwas_prior_odds": gwas_prior_odds})

                    if learn_window:

                        use_logistic_window_function = False
                        if use_logistic_window_function:

                            #run this a few times
                            num_samples = 5
                            window_fun_slope = 0
                            window_fun_intercept = 0

                            for i in range(num_samples):
                                sample = np.random.random(len(closest_dist_Y)) < closest_dist_Y
                                closest_dist_Y_sample = copy.copy(closest_dist_Y)
                                closest_dist_Y_sample[sample > closest_dist_Y] = 1
                                closest_dist_Y_sample[sample <= closest_dist_Y] = 0

                                (cur_window_fun_slope, se, z, p, se_inflation_factor, cur_window_fun_intercept, diverged) = self._compute_logistic_beta_tildes(closest_dist_X[:,np.newaxis], closest_dist_Y_sample, 1, 0, resid_correlation_matrix=None, convert_to_dichotomous=False, log_fun=lambda x, y=0: 1)
                                window_fun_slope += cur_window_fun_slope
                                window_fun_intercept += cur_window_fun_intercept

                            window_fun_slope /= num_samples
                            window_fun_intercept /= num_samples
                        else:

                            mean_closest_dist_X = np.mean(closest_dist_X[closest_dist_Y == closest_gene_prob])
                            mean_non_closest_dist_X = np.mean(closest_dist_X[closest_dist_Y != closest_gene_prob])
                            mean_non_closest_dist_Y = np.mean(closest_dist_Y[closest_dist_Y != closest_gene_prob])
                            window_fun_slope  = (np.log(closest_gene_prob / (1 - closest_gene_prob)) - np.log(mean_non_closest_dist_Y / (1 - mean_non_closest_dist_Y))) / (mean_closest_dist_X - mean_non_closest_dist_X)
                            window_fun_intercept = np.log(closest_gene_prob / (1 - closest_gene_prob)) - window_fun_slope * mean_closest_dist_X

                        if window_fun_slope >= 0:
                            warn("Could not fit decaying linear window function slope for max-closest-gene-dist=%.4g and closest-gene_prob=%.4g; using default" % (max_closest_gene_dist, closest_gene_prob))
                            window_fun_slope = -6.983e-06
                            window_fun_intercept = -1.934

                        log("Fit function %.4g * x + %.4g for closest gene probability" % (window_fun_slope, window_fun_intercept))

                    else:
                        if max_closest_gene_dist < 3e5:
                            window_fun_slope = -5.086e-05
                            window_fun_intercept = 2.988
                        else:
                            window_fun_slope = -5.152e-05
                            window_fun_intercept = 4.854
                        log("Using %.4g * x + %.4g for closest gene probability" % (window_fun_slope, window_fun_intercept))

                    self._record_params({"window_fun_slope": window_fun_slope, "window_fun_intercept": window_fun_intercept})

            #now iterate through all significant variants

            log("Done reading --gwas-in", DEBUG)

            if len(self.huge_signals) == 0:
                 bail("Didn't read in any SNPs for HuGE scores")


            exomes_positive_controls_prior_log_bf = None

            if self.genes is not None:
                genes = self.genes
                gene_to_ind = self.gene_to_ind
            else:
                genes = list(gene_to_chrom.keys())
                gene_to_ind = self._construct_map_to_ind(genes)

            #need to remap the indices
            extra_genes = []
            extra_gene_to_ind = {}
            for gene_prob_rows_to_process in [gene_prob_rows, gene_prob_rows_detect]:
                for i in range(len(gene_prob_rows_to_process)):
                    cur_gene = gene_prob_genes[gene_prob_rows_to_process[i]]

                    if cur_gene in gene_to_ind:
                        new_ind = gene_to_ind[cur_gene]
                    elif cur_gene in extra_gene_to_ind:
                        new_ind = extra_gene_to_ind[cur_gene]
                    else:
                        new_ind = len(extra_genes) + len(genes)
                        extra_genes.append(cur_gene)
                        extra_gene_to_ind[cur_gene] = new_ind
                    gene_prob_rows_to_process[i] = new_ind

            #add in any genes that were missed
            for cur_gene in list(gene_to_chrom.keys()) + gene_prob_genes:
                if cur_gene not in gene_to_ind and cur_gene not in extra_gene_to_ind:
                    new_ind = len(extra_genes) + len(genes)
                    extra_genes.append(cur_gene)
                    extra_gene_to_ind[cur_gene] = new_ind

            gene_prob_gene_list = genes + extra_genes

            if self.gene_covariates is not None:

                #sort the covariate file; initially populate it with mean value in case some genes are missing from it

                sorted_gene_covariates = np.tile(np.nanmean(self.gene_covariates, axis=0), len(gene_prob_gene_list)).reshape((len(gene_prob_gene_list), self.gene_covariates.shape[1]))

                for i in range(len(gene_covariate_genes)):
                    cur_gene = gene_covariate_genes[i]
                    assert(cur_gene in gene_to_ind or cur_gene in extra_gene_to_ind)

                    if cur_gene in gene_to_ind:
                        new_ind = gene_to_ind[cur_gene]
                    elif cur_gene in extra_gene_to_ind:
                        new_ind = extra_gene_to_ind[cur_gene]
                    noninf_mask = ~np.isnan(self.gene_covariates[i,:])
                    sorted_gene_covariates[new_ind,noninf_mask] = self.gene_covariates[i,noninf_mask]

                self.gene_covariates = sorted_gene_covariates


            if self.Y_exomes is not None:
                assert(len(genes) == len(self.Y_exomes))
                exomes_positive_controls_prior_log_bf = np.append(self.Y_exomes, np.zeros(len(extra_genes)))
            if self.Y_positive_controls is not None:
                assert(len(genes) == len(self.Y_positive_controls))
                positive_controls_prior_log_bf = np.append(self.Y_positive_controls, np.zeros(len(extra_genes)))
                if exomes_positive_controls_prior_log_bf is None:
                    exomes_positive_controls_prior_log_bf = positive_controls_prior_log_bf
                else:
                    exomes_positive_controls_prior_log_bf += positive_controls_prior_log_bf


            #add in the extra genes

            #this is the normalizing constant between huge_signal_bfs and the BFs
            #PO = BF * huge_signal_posteriors
            self.huge_signal_posteriors = np.array(self.huge_signal_posteriors)
            self.huge_signal_posteriors_for_regression = np.array(self.huge_signal_posteriors_for_regression)
            self.huge_signal_max_closest_gene_prob = max_closest_gene_prob
            self.huge_cap_region_posterior = cap_region_posterior
            self.huge_scale_region_posterior = scale_region_posterior
            self.huge_phantom_region_posterior = phantom_region_posterior
            self.huge_allow_evidence_of_absence = allow_evidence_of_absence

            #from Maller et al, these are proportional to BFs (but not necessarily equal)
            self.huge_signal_bfs = sparse.csc_matrix((gene_bf_data, (gene_prob_rows, gene_prob_cols)), shape=(len(gene_prob_gene_list), gene_prob_col_num))
            self.huge_signal_bfs_for_regression = sparse.csc_matrix((gene_bf_data_detect, (gene_prob_rows_detect, gene_prob_cols_detect)), shape=(len(gene_prob_gene_list), gene_prob_col_num))

            self.huge_signal_sum_gene_cond_probabilities = np.array(self.huge_signal_sum_gene_cond_probabilities)
            self.huge_signal_sum_gene_cond_probabilities_for_regression = np.array(self.huge_signal_sum_gene_cond_probabilities_for_regression)
            self.huge_signal_mean_gene_pos = np.array(self.huge_signal_mean_gene_pos)
            self.huge_signal_mean_gene_pos_for_regression = np.array(self.huge_signal_mean_gene_pos_for_regression)


            #construct the matrix

            #don't do the correction here (will do it outside)
            (huge_results, huge_results_uncorrected, absent_genes, absent_log_bf) = self._distill_huge_signal_bfs(self.huge_signal_bfs, self.huge_signal_posteriors, self.huge_signal_sum_gene_cond_probabilities, self.huge_signal_mean_gene_pos, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, None, None, None, None, None, gene_prob_gene_list, total_genes=self.genes, rel_prior_log_bf=exomes_positive_controls_prior_log_bf)

            (huge_results_for_regression, huge_results_uncorrected_for_regression, absent_genes_for_regression, absent_log_bf_for_regression) = self._distill_huge_signal_bfs(self.huge_signal_bfs_for_regression, self.huge_signal_posteriors_for_regression, self.huge_signal_sum_gene_cond_probabilities_for_regression, self.huge_signal_mean_gene_pos_for_regression, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, None, None, None, None, None, gene_prob_gene_list, total_genes=self.genes, rel_prior_log_bf=exomes_positive_controls_prior_log_bf)

            if self.genes is not None:
                gene_bf = np.array([np.nan] * len(self.genes))
                gene_bf_for_regression = np.array([np.nan] * len(self.genes))
            else:
                gene_bf = np.array([])
                gene_bf_for_regression = np.array([])

            extra_gene_bf = []
            extra_gene_bf_for_regression = []
            extra_genes = []
            self.gene_to_gwas_huge_score = {}
            self.gene_to_gwas_huge_score_uncorrected = {}

            for i in range(len(gene_prob_gene_list)):
                gene = gene_prob_gene_list[i]
                bf = huge_results[i]
                bf_for_regression = huge_results_for_regression[i]
                bf_uncorrected = huge_results_uncorrected[i]
                self.gene_to_gwas_huge_score[gene] = bf
                self.gene_to_gwas_huge_score_uncorrected[gene] = bf_uncorrected
                if self.genes is not None and gene in self.gene_to_ind:
                    assert(self.gene_to_ind[gene] == i)
                    gene_bf[self.gene_to_ind[gene]] = bf
                    gene_bf_for_regression[self.gene_to_ind[gene]] = bf_for_regression
                else:
                    extra_gene_bf.append(bf)
                    extra_gene_bf_for_regression.append(bf_for_regression)
                    extra_genes.append(gene)
            for gene in absent_genes:
                bf = absent_log_bf
                bf_for_regression = absent_log_bf_for_regression
                self.gene_to_gwas_huge_score[gene] = bf
                self.gene_to_gwas_huge_score_uncorrected[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                    gene_bf_for_regression[self.gene_to_ind[gene]] = bf_for_regression
                else:
                    extra_gene_bf.append(bf)
                    extra_gene_bf_for_regression.append(bf_for_regression)
                    extra_genes.append(gene)
            extra_gene_bf = np.array(extra_gene_bf)

            self.combine_huge_scores()

            total_gene_bfs = np.append(gene_bf_for_regression, extra_gene_bf_for_regression)
            number_same = np.max(np.unique(np.append(gene_bf_for_regression, extra_gene_bf_for_regression), return_counts=True)[1])
            fraction_same = number_same / float(len(total_gene_bfs))
            if fraction_same > 0.4:
                log("Had %d out of %d genes with the the same huge scores; too few genes to run regressions to learn confounder corrections" % (number_same, len(total_gene_bfs)))
                self.huge_sparse_mode = True

            return (gene_bf, extra_genes, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression)

    def calculate_huge_scores_exomes(self, exomes_in, exomes_gene_col=None, exomes_p_col=None, exomes_beta_col=None, exomes_se_col=None, exomes_n_col=None, exomes_n=None, exomes_units=None, allelic_var=0.36, exomes_low_p=2.5e-6, exomes_high_p=0.05, exomes_low_p_posterior=0.95, exomes_high_p_posterior=0.10, hold_out_chrom=None, gene_loc_file=None, **kwargs):

        if exomes_in is None:
            bail("Require --exomes-in for this operation")

        if hold_out_chrom is not None and self.gene_to_chrom is None:
            (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

        self._record_params({"exomes_low_p": exomes_low_p, "exomes_high_p": exomes_high_p, "exomes_low_p_posterior": exomes_low_p_posterior, "exomes_high_p_posterior": exomes_high_p_posterior})

        if exomes_gene_col is None:
            need_columns = True

        has_se = exomes_se_col is not None or exomes_n_col is not None or exomes_n is not None
        if exomes_gene_col is not None and ((exomes_p_col is not None and exomes_beta_col is not None) or (exomes_p_col is not None and has_se) or (exomes_beta_col is not None and has_se)):
            need_columns = False
        else:
            need_columns = True

        if need_columns:
            (possible_gene_id_cols, possible_var_id_cols, possible_chrom_cols, possible_pos_cols, possible_locus_cols, possible_p_cols, possible_beta_cols, possible_se_cols, possible_freq_cols, possible_n_cols, header) = self._determine_columns(exomes_in)

            #now recompute
            if exomes_gene_col is None:
                if len(possible_gene_id_cols) == 1:
                    exomes_gene_col = possible_gene_id_cols[0]
                    log("Using %s for gene_id column; change with --exomes-gene-col if incorrect" % exomes_gene_col)
                else:
                    bail("Could not determine gene_id column from header %s; specify with --exomes-gene-col" % header)

            if exomes_p_col is None:
                if len(possible_p_cols) == 1:
                    exomes_p_col = possible_p_cols[0]
                    log("Using %s for p column; change with --exomes-p-col if incorrect" % exomes_p_col)
                else:
                    log("Could not determine p column from header %s; if desired specify with --exomes-p-col" % header)
            if exomes_se_col is None:
                if len(possible_se_cols) == 1:
                    exomes_se_col = possible_se_cols[0]
                    log("Using %s for se column; change with --exomes-se-col if incorrect" % exomes_se_col)
                else:
                    log("Could not determine se column from header %s; if desired specify with --exomes-se-col" % header)
            if exomes_beta_col is None:
                if len(possible_beta_cols) == 1:
                    exomes_beta_col = possible_beta_cols[0]
                    log("Using %s for beta column; change with --exomes-beta-col if incorrect" % exomes_beta_col)
                else:
                    log("Could not determine beta column from header %s; if desired specify with --exomes-beta-col" % header)

            if exomes_n_col is None:
                if len(possible_n_cols) == 1:
                    exomes_n_col = possible_n_cols[0]
                    log("Using %s for N column; change with --exomes-n-col if incorrect" % exomes_n_col)
                else:
                    log("Could not determine N column from header %s; if desired specify with --exomes-n-col" % header)

            has_se = exomes_se_col is not None or exomes_n_col is not None or exomes_n is not None
            if (exomes_p_col is not None and exomes_beta_col is not None) or (exomes_p_col is not None and has_se) or (exomes_beta_col is not None and has_se):
                pass
            else:
                bail("Require information about at least two of p-value, se, and beta; specify with --exomes-p-col, --exomes-beta-col, and --exomes-se-col")

        (allelic_var_k, exomes_prior_odds) = self.compute_allelic_var_and_prior(exomes_high_p, exomes_high_p_posterior, exomes_low_p, exomes_low_p_posterior)

        self._record_params({"exomes_allelic_var_k": allelic_var_k, "exomes_prior_odds": exomes_prior_odds})

        log("Using exomes k=%.3g, po=%.3g" % (allelic_var_k, exomes_prior_odds))

        log("Calculating exomes HuGE scores")

        log("Reading --exomes-in file %s" % exomes_in, INFO)

        seen_genes = set()
        genes = []
        gene_ps = []
        gene_betas = []
        gene_ses = []

        with open_gz(exomes_in) as exomes_fh:
            header_cols = exomes_fh.readline().strip().split()
            gene_col = self._get_col(exomes_gene_col, header_cols)

            p_col = None
            if exomes_p_col is not None:
                p_col = self._get_col(exomes_p_col, header_cols)

            beta_col = None
            if exomes_beta_col is not None:
                beta_col = self._get_col(exomes_beta_col, header_cols)

            n_col = None
            se_col = None
            if exomes_n_col is not None:
                n_col = self._get_col(exomes_n_col, header_cols)
            if exomes_se_col is not None:
                se_col = self._get_col(exomes_se_col, header_cols)
            
            chrom_pos_p_se = {}

            #read in the exomes associations
            total_num_genes = 0

            for line in exomes_fh:

                cols = line.strip().split()
                if gene_col > len(cols) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)) or (beta_col is not None and beta_col > len(cols)) or (n_col is not None and n_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[gene_col]

                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                if hold_out_chrom is not None and gene in self.gene_to_chrom and self.gene_to_chrom[gene] == hold_out_chrom:
                    continue

                p = None
                beta = None
                se = None
                
                if p_col is not None:
                    try:
                        p = float(cols[p_col])
                    except ValueError:
                        if not cols[p_col] == "NA":
                            warn("Skipping unconvertible p value %s" % (cols[p_col]))
                        continue

                    min_p = 1e-250
                    if p < min_p:
                        p = min_p

                    if p <= 0 or p > 1:
                        warn("Skipping invalid p value %s" % (p))
                        continue

                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                    except ValueError:
                        if not cols[beta_col] == "NA":
                            warn("Skipping unconvertible beta value %s" % (cols[beta_col]))
                        continue

                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            warn("Skipping unconvertible se value %s" % (cols[se_col]))
                        continue
                elif n_col is not None:
                    try:
                        n = float(cols[n_col])
                    except ValueError:
                        if not cols[n_col] == "NA":
                            warn("Skipping unconvertible n value %s" % (cols[n_col]))
                        continue
                        
                    if n <= 0:
                        warn("Skipping invalid N value %s" % (n))
                        continue
                    se = 1 / np.sqrt(n)
                elif exomes_n is not None:
                    if exomes_n <= 0:
                        bail("Invalid exomes-n value: %s" % (exomesa_n))
                        continue
                    n = exomes_n
                    se = 1 / np.sqrt(n)

                total_num_genes += 1

                if gene in seen_genes:
                    warn("Gene %s has been seen before; skipping all but first occurrence" % gene)
                    continue
                
                seen_genes.add(gene)
                genes.append(gene)
                gene_ps.append(p)
                gene_betas.append(beta)
                gene_ses.append(se)

            #determine scale_factor
          
            gene_ps = np.array(gene_ps, dtype=float)
            gene_betas = np.array(gene_betas, dtype=float)
            gene_ses = np.array(gene_ses, dtype=float)

            (gene_ps, gene_betas, gene_ses) = self._complete_p_beta_se(gene_ps, gene_betas, gene_ses)

            gene_zs = gene_betas / gene_ses

            gene_ses2 = np.square(gene_ses)

            log("Done reading --exomes-in", DEBUG)

            #adjust units of beta if beta column was passed in
            #if exomes_units is not None:
            #    allelic_var *= np.square(exomes_units)
            #    log("Scaling allelic variance %.3g-fold to be %.4g" % (np.square(exomes_units), allelic_var))
            #else:
            #    #get the empirical variance of the betas for variants in a range of p=0.05
            #    p05mask = np.abs(np.abs(gene_betas/gene_ses) - 1.95) <= 0.07
            #    if np.sum(p05mask) > 100:
            #        emp_beta_var = np.mean(np.square(gene_betas[p05mask]) - gene_ses2[p05mask])
            #        #this is roughly what we observe for a dichotomous trait in this range. Larger than for gwas by about 10x
            #        ref_emp_beta_var = 0.1
            #        if emp_beta_var > 0 and (emp_beta_var / ref_emp_beta_var > 5 or emp_beta_var / ref_emp_beta_var < 0.2):
            #            allelic_var *= (emp_beta_var / 0.1)
            #            log("Scaling allelic variance %.3g-fold to be %.4g" % (emp_beta_var / 0.1, allelic_var))

            #gene_log_bfs = np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(gene_zs) * allelic_var_k / (1 + allelic_var_k)

            gene_log_bfs = -np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(gene_zs) * allelic_var_k / (1 + allelic_var_k)

            max_log = 15
            gene_log_bfs[gene_log_bfs > max_log] = max_log

            #set lower bound not here but below; otherwise it gets inflated above background
            #gene_log_bfs[gene_log_bfs < 0] = 0

            gene_post = np.exp(gene_log_bfs + np.log(exomes_prior_odds))
            gene_probs = gene_post / (gene_post + 1)
            gene_probs[gene_probs < self.background_prior] = self.background_prior

            #gene_probs_sum = np.sum(gene_probs)

            absent_genes = set()
            if self.genes is not None:
                #have to account for these
                absent_genes = set(self.genes) - set(genes)
            #gene_probs_sum += self.background_prior * len(absent_genes)

            norm_constant = 1
            #norm_constant = (self.background_prior * (len(gene_probs) + len(absent_genes))) / gene_probs_sum
            #need at least 1000 genes
            #if len(gene_probs) < 1000:
            #    norm_constant = 1
            #gene_probs *= norm_constant


            gene_log_bfs = np.log(gene_probs / (1 - gene_probs)) - self.background_log_bf

            absent_prob = self.background_prior * norm_constant
            absent_log_bf = np.log(absent_prob / (1 - absent_prob)) - self.background_log_bf

            if self.genes is not None:
                gene_bf = np.array([np.nan] * len(self.genes))
            else:
                gene_bf = np.array([])

            extra_gene_bf = []
            extra_genes = []
            self.gene_to_exomes_huge_score = {}

            for i in range(len(genes)):
                gene = genes[i]
                bf = gene_log_bfs[i]
                self.gene_to_exomes_huge_score[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            for gene in absent_genes:
                bf = absent_log_bf
                self.gene_to_exomes_huge_score[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            extra_gene_bf = np.array(extra_gene_bf)

            self.combine_huge_scores()
            return (gene_bf, extra_genes, extra_gene_bf)

    def read_positive_controls(self, positive_controls_in, positive_controls_id_col=None, positive_controls_prob_col=None, positive_controls_default_prob=0.95, positive_controls_has_header=True, positive_controls_list=None, positive_controls_all_in=None, positive_controls_all_id_col=None, positive_controls_all_has_header=True, hold_out_chrom=None, gene_loc_file=None, **kwargs):
        if positive_controls_in is None and positive_controls_list is None:
            bail("Require --positive-controls-in or --positive-controls-list for this operation")

        if hold_out_chrom is not None and self.gene_to_chrom is None:
            (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

        if positive_controls_default_prob >= 1:
            positive_controls_default_prob = 0.99
        if positive_controls_default_prob <= 0:
            positive_controls_default_prob = 0.01

        self.gene_to_positive_controls = {}
        if positive_controls_list is not None:
            for gene in positive_controls_list:
                self.gene_to_positive_controls[gene] = np.log(positive_controls_default_prob / (1 - positive_controls_default_prob)) - self.background_log_bf

        positive_control_files = []
        if positive_controls_in is not None:
            positive_control_files.append((positive_controls_in, positive_controls_id_col, positive_controls_prob_col, positive_controls_default_prob, positive_controls_has_header))
        if positive_controls_all_in is not None:
            positive_control_files.append((positive_controls_all_in, positive_controls_all_id_col, None, self.background_prior, positive_controls_all_has_header))

        for (cur_positive_controls_in, cur_id_col, cur_prob_col, default_prob, has_header) in positive_control_files:
            log("Reading --positive-controls-in file %s" % cur_positive_controls_in, INFO)

            with open_gz(cur_positive_controls_in) as positive_controls_fh:
                id_col = 0
                prob_col = None
                seen_header = False
                for line in positive_controls_fh:
                    cols = line.strip().split()
                    if not seen_header:
                        seen_header = True
                        if has_header or len(cols) > 1:
                            if len(cols) > 1 and cur_id_col is None:
                                bail("--positive-controls-id-col required for positive control files with more than one column")
                            elif cur_id_col is not None:
                                id_col = self._get_col(cur_id_col, cols)

                            if cur_prob_col is not None:
                                prob_col = self._get_col(cur_prob_col, cols)

                            if has_header and cur_id_col is not None:
                                continue

                    if id_col >= len(cols):
                        warn("Skipping due to too few columns in line: %s" % line)
                        continue

                    gene = cols[id_col]

                    if self.gene_label_map is not None and gene in self.gene_label_map:
                        gene = self.gene_label_map[gene]

                    if hold_out_chrom is not None and gene in self.gene_to_chrom and self.gene_to_chrom[gene] == hold_out_chrom:
                        continue

                    prob = default_prob
                    if prob_col is not None and prob_col >= len(cols):
                        warn("Skipping due to too few columns in line: %s" % line)
                        continue

                    if prob_col is not None:
                        try:
                            prob = float(cols[prob_col])
                            if prob <= 0 or prob >= 1:
                                warn("Probabilities must be in (0,1); observed %s for %s" % (prob, gene))
                                continue
                        except ValueError:
                            if not cols[prob_col] == "NA":
                                warn("Skipping unconvertible prob value %s for gene %s" % (cols[prob_col], gene))
                            continue

                    max_prob = 0.99
                    if prob > max_prob:
                        prob = max_prob
                    log_bf = np.log(prob / (1 - prob)) - self.background_log_bf
                    if gene not in self.gene_to_positive_controls:
                        self.gene_to_positive_controls[gene] = log_bf

        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        positive_controls = np.array([np.nan] * len(genes))
        
        extra_positive_controls = []
        extra_genes = []
        for gene in self.gene_to_positive_controls:
            log_bf = self.gene_to_positive_controls[gene]
            if gene in gene_to_ind:
                positive_controls[gene_to_ind[gene]] = log_bf
            else:
                extra_positive_controls.append(log_bf)
                extra_genes.append(gene)

        return (positive_controls, extra_genes, np.array(extra_positive_controls))


    def compute_allelic_var_and_prior(self, high_p, high_p_posterior, low_p, low_p_posterior):

        if high_p < low_p:
            warn("Swapping high_p and low_p")
            temp = high_p
            high_p = low_p
            low_p = temp

        if high_p == low_p:
            high_p = low_p * 2

        if high_p_posterior >= 1:
            po_high = 0.99/0.01
        elif high_p_posterior <=0 :
            po_high = 0.001/0.999
        else:
            po_high = high_p_posterior / (1 - high_p_posterior)

        if low_p_posterior >= 1:
            po_low = 0.99/0.01
        elif low_p_posterior <=0 :
            po_low = 0.001/0.999
        else:
            po_low = low_p_posterior / (1 - low_p_posterior)

        z_high = np.abs(scipy.stats.norm.ppf(high_p/2))
        z_low = np.abs(scipy.stats.norm.ppf(low_p/2))
        ratio = po_low / po_high

        allelic_var_k = 2 * np.log(ratio) / (np.square(z_low) - np.square(z_high))

        if allelic_var_k > 1:
            #reset high_p_posterior
            max_allelic_var_k = 0.99;
            po_high = po_low / np.exp(max_allelic_var_k * (np.square(z_low) - np.square(z_high)) / 2)
            log("allelic_var_k overflow; adjusting --high-p-posterior to %.4g" % (po_high/(1+po_high)))
            ratio = po_low / po_high
            allelic_var_k = 2 * np.log(ratio) / (np.square(z_low) - np.square(z_high))

        allelic_var_k = allelic_var_k / (1 - allelic_var_k)

        prior_odds = po_low / (np.sqrt(1 / (1 + allelic_var_k)) * np.exp(0.5 * np.square(z_low) * (allelic_var_k / (1 + allelic_var_k))))
        
        return (allelic_var_k, prior_odds)


    def combine_huge_scores(self):
        #combine the huge scores if needed
        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            self.gene_to_huge_score = {}
            genes = list(set().union(self.gene_to_gwas_huge_score, self.gene_to_exomes_huge_score))
            for gene in genes:
                self.gene_to_huge_score[gene] = 0
                if gene in self.gene_to_gwas_huge_score:
                    self.gene_to_huge_score[gene] += self.gene_to_gwas_huge_score[gene]
                if gene in self.gene_to_exomes_huge_score:
                    self.gene_to_huge_score[gene] += self.gene_to_exomes_huge_score[gene]

    def read_gene_set_statistics(self, stats_in, stats_id_col=None, stats_exp_beta_tilde_col=None, stats_beta_tilde_col=None, stats_p_col=None, stats_se_col=None, stats_beta_col=None, stats_beta_uncorrected_col=None, ignore_negative_exp_beta=False, max_gene_set_p=None, min_gene_set_beta=None, min_gene_set_beta_uncorrected=None, return_only_ids=False):

        if stats_in is None:
            bail("Require --stats-in for this operation")

        log("Reading --stats-in file %s" % stats_in, INFO)
        subset_mask = None
        need_to_take_log = False

        read_ids = set()

        with open_gz(stats_in) as stats_fh:
            header_cols = stats_fh.readline().strip().split()
            id_col = self._get_col(stats_id_col, header_cols)
            beta_tilde_col = None

            if stats_beta_tilde_col is not None:
                beta_tilde_col = self._get_col(stats_beta_tilde_col, header_cols, False)
            if beta_tilde_col is not None:
                log("Using col %s for beta_tilde values" % stats_beta_tilde_col)
            elif stats_exp_beta_tilde_col is not None:
                beta_tilde_col = self._get_col(stats_exp_beta_tilde_col, header_cols)
                need_to_take_log = True
                if beta_tilde_col is not None:
                    log("Using %s for exp(beta_tilde) values" % stats_exp_beta_tilde_col)
                else:
                    bail("Could not find beta_tilde column %s or %s in header: %s" % (stats_beta_tilde_col, stats_exp_beta_tilde_col, "\t".join(header_cols)))

            p_col = None
            if stats_p_col is not None:
                p_col = self._get_col(stats_p_col, header_cols, False)            

            se_col = None
            if stats_se_col is not None:
                se_col = self._get_col(stats_se_col, header_cols, False)            

            beta_col = None
            if stats_beta_col is not None:
                beta_col = self._get_col(stats_beta_col, header_cols, True)
            else:
                beta_col = self._get_col("beta", header_cols, False)
                
            beta_uncorrected_col = None
            if stats_beta_uncorrected_col is not None:
                beta_uncorrected_col = self._get_col(stats_beta_uncorrected_col, header_cols, True)
            else:
                beta_uncorrected_col = self._get_col("beta_uncorrected", header_cols, False)

            if se_col is None and p_col is None and beta_tilde_col is None and beta_col is None and beta_uncorrected_col is None:
                bail("Require at least something to read from --gene-set-stats-in")

            if not return_only_ids:

                if self.gene_sets is not None:
                    if beta_tilde_col is not None:
                        self.beta_tildes = np.zeros(len(self.gene_sets))
                    if p_col is not None or se_col is not None:
                        self.p_values = np.zeros(len(self.gene_sets))
                        self.ses = np.zeros(len(self.gene_sets))
                        self.z_scores = np.zeros(len(self.gene_sets))
                    if beta_col is not None:
                        self.betas = np.zeros(len(self.gene_sets))
                    if beta_uncorrected_col is not None:
                        self.betas_uncorrected = np.zeros(len(self.gene_sets))

                    subset_mask = np.array([False] * len(self.gene_sets))
                else:
                    if beta_tilde_col is not None:
                        self.beta_tildes = []
                    if p_col is not None or se_col is not None:
                        self.p_values = []
                        self.ses = []
                        self.z_scores = []
                    if beta_col is not None:
                        self.betas = []
                    if beta_uncorrected_col is not None:
                        self.betas_uncorrected = []

            gene_sets = []
            gene_set_to_ind = {}

            ignored = 0

            already_seen = 0

            for line in stats_fh:
                beta_tilde = None
                alpha_tilde = None
                p = None
                se = None
                z = None
                beta = None
                beta_uncorrected = None

                cols = line.strip().split()
                if id_col > len(cols) or (beta_tilde_col is not None and beta_tilde_col > len(cols)) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene_set = cols[id_col]
                if gene_set in gene_set_to_ind:
                    warn("Already seen gene set %s; only considering first instance" % (gene_set))
                    continue

                if beta_tilde_col is not None:
                    try:
                        beta_tilde = float(cols[beta_tilde_col])
                    except ValueError:
                        if not cols[beta_tilde_col] == "NA":
                            warn("Skipping unconvertible beta_tilde value %s for gene_set %s" % (cols[beta_tilde_col], gene_set))
                        continue

                    if need_to_take_log:
                        if beta_tilde < 0:
                            if ignore_negative_exp_beta:
                                continue
                            bail("Exp(beta) value %s for gene set %s is < 0; did you mean to specify --stats-beta-col? Otherwise, specify --ignore-negative-exp-beta to ignore these" % (beta_tilde, gene_set))
                        beta_tilde = np.log(beta_tilde)
                    alpha_tilde = 0

                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            warn("Skipping unconvertible se value %s for gene_set %s" % (cols[se_col], gene_set))
                        continue

                    if beta_tilde_col is not None:
                        z = beta_tilde / se
                        p = 2*scipy.stats.norm.cdf(-np.abs(z))
                        if max_gene_set_p is not None and p > max_gene_set_p:
                            continue
                elif p_col is not None:
                    try:
                        p = float(cols[p_col])
                        if max_gene_set_p is not None and p > max_gene_set_p:
                            continue
                    except ValueError:
                        if not cols[p_col] == "NA":
                            warn("Skipping unconvertible p value %s for gene_set %s" % (cols[p_col], gene_set))
                        continue

                    z = np.abs(scipy.stats.norm.ppf(p/2))
                    if z == 0:
                        warn("Skipping gene_set %s due to 0 z-score" % (gene_set))
                        continue

                    if beta_tilde_col is not None:
                        se = np.abs(beta_tilde) / z

                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                        if min_gene_set_beta is not None and beta < min_gene_set_beta:
                            continue

                    except ValueError:
                        if not cols[beta_col] == "NA":
                            warn("Skipping unconvertible beta value %s for gene_set %s" % (cols[beta_col], gene_set))
                        continue

                if beta_uncorrected_col is not None:
                    try:
                        beta_uncorrected = float(cols[beta_uncorrected_col])
                        if min_gene_set_beta_uncorrected is not None and beta_uncorrected < min_gene_set_beta_uncorrected:
                            continue

                    except ValueError:
                        if not cols[beta_uncorrected_col] == "NA":
                            warn("Skipping unconvertible beta_uncorrected value %s for gene_set %s" % (cols[beta_uncorrected_col], gene_set))
                        continue


                gene_set_ind = None

                if self.gene_sets is not None:
                    if gene_set not in self.gene_set_to_ind:
                        ignored += 1
                        continue

                    if return_only_ids:
                        read_ids.add(gene_set)
                        continue

                    gene_set_ind = self.gene_set_to_ind[gene_set]
                    if gene_set_ind is not None:
                        if beta_tilde_col is not None:
                            self.beta_tildes[gene_set_ind] = beta_tilde * self.scale_factors[gene_set_ind]
                        if p_col is not None or se_col is not None:
                            self.p_values[gene_set_ind] = p
                            self.z_scores[gene_set_ind] = z
                            self.ses[gene_set_ind] = se * self.scale_factors[gene_set_ind]
                        if beta_col is not None:
                            self.betas[gene_set_ind] = beta * self.scale_factors[gene_set_ind]
                        if beta_uncorrected_col is not None:
                            self.betas_uncorrected[gene_set_ind] = beta_uncorrected * self.scale_factors[gene_set_ind]
                        subset_mask[gene_set_ind] = True
                else:
                    if return_only_ids:
                        read_ids.add(gene_set)
                        continue

                    bail("Not yet implemented this -- no way to convert external beta tilde units reading in into internal units")
                    if beta_tilde_col is not None:
                        self.beta_tildes.append(beta_tilde)
                    if p_col is not None or se_col is not None:
                        self.p_values.append(p)
                        self.z_scores.append(z)
                        self.ses.append(se)
                    if beta_col is not None:
                        self.betas.append(beta)
                    if beta_uncorrected_col is not None:
                        self.betas_uncorrected.append(beta_uncorrected)

                    #store these in all cases to be able to check for duplicate gene sets in the input
                    gene_set_to_ind[gene_set] = len(gene_sets)
                    gene_sets.append(gene_set)

            log("Done reading --stats-in-file", DEBUG)

        if return_only_ids:
            return read_ids

        if self.gene_sets is not None:
            log("Subsetting matrices", DEBUG)
            #need to subset existing matrices
            if ignored > 0:
                warn("Ignored %s values from --stats-in file because absent from previously loaded files" % ignored)
            if sum(subset_mask) != len(subset_mask):
                warn("Excluding %s values from previously loaded files because absent from --stats-in file" % (len(subset_mask) - sum(subset_mask)))
                if self.beta_tildes is not None and not need_to_take_log and sum(self.beta_tildes < 0) == 0:
                    warn("All beta_tilde values are positive. Are you sure that the values in column %s are not exp(beta_tilde)?" % stats_beta_col)
                self.subset_gene_sets(subset_mask, keep_missing=True)
            log("Done subsetting matrices", DEBUG)
        else:
            self.X_orig_missing_gene_sets = None
            self.mean_shifts_missing = None
            self.scale_factors_missing = None
            self.is_dense_gene_set_missing = None
            self.ps_missing = None
            self.sigma2s_missing = None

            self.beta_tildes_missing = None
            self.p_values_missing = None
            self.ses_missing = None
            self.z_scores_missing = None

            self.beta_tildes = np.array(self.beta_tildes)
            self.p_values = np.array(self.p_values)
            self.z_scores = np.array(self.z_scores)
            self.ses = np.array(self.ses)
            self.gene_sets = gene_sets
            self.gene_set_to_ind = gene_set_to_ind

            if beta_col is not None:
                self.betas = np.array(self.betas)
            if beta_uncorrected_col is not None:
                self.betas_uncorrected = np.array(self.betas_uncorrected)

            self.total_qc_metrics_missing = None
            self.mean_qc_metrics_missing = None

        #self.max_gene_set_p = max_gene_set_p
        #self.is_logistic = False
        #make sure we are doing the normalization
        self._set_X(self.X_orig, self.genes, self.gene_sets, skip_N=True)


    def read_gene_set_phewas_statistics(self, stats_in, stats_id_col=None, stats_pheno_col=None, stats_beta_col=None, stats_beta_uncorrected_col=None, min_gene_set_beta=None, min_gene_set_beta_uncorrected=None, update_X=False, phenos_to_match=None, return_only_ids=False):

        if stats_in is None:
            bail("Require --gene-set-stats-in or --gene-set-phewas-stats-in for this operation")

        log("Reading --gene-set-phewas-stats-in file %s" % stats_in, INFO)
        subset_mask = None
        need_to_take_log = False

        read_ids = set()

        with open_gz(stats_in) as stats_fh:
            header_cols = stats_fh.readline().strip().split()
            id_col = self._get_col(stats_id_col, header_cols)
            pheno_col = self._get_col(stats_pheno_col, header_cols)

            beta_col = None
            if stats_beta_col is not None:
                beta_col = self._get_col(stats_beta_col, header_cols, True)
            else:
                beta_col = self._get_col("beta", header_cols, False)
                
            beta_uncorrected_col = None
            if stats_beta_uncorrected_col is not None:
                beta_uncorrected_col = self._get_col(stats_beta_uncorrected_col, header_cols, True)
            else:
                beta_uncorrected_col = self._get_col("beta_uncorrected", header_cols, False)

            if beta_col is None and beta_uncorrected_col is None:
                bail("Require at least something to read from --gene-set-stats-in")

            if self.gene_sets is not None:
                subset_mask = np.array([False] * len(self.gene_sets))

            gene_sets = []
            gene_set_to_ind = {}

            phenos = []
            pheno_to_ind = {}

            ignored = 0

            betas = []
            betas_uncorrected = []
            row = []
            col = []

            for line in stats_fh:
                beta = None
                beta_uncorrected = None

                cols = line.strip().split()
                if id_col > len(cols) or pheno_col > len(cols) or (beta_col is not None and beta_col > len(cols)) or (beta_uncorrected_col is not None and beta_uncorrected_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene_set = cols[id_col]
                pheno = cols[pheno_col]

                if phenos_to_match is not None and pheno not in phenos_to_match:
                    continue

                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                        if min_gene_set_beta is not None and beta < min_gene_set_beta:
                            continue

                    except ValueError:
                        if not cols[beta_col] == "NA":
                            warn("Skipping unconvertible beta value %s for gene_set %s" % (cols[beta_col], gene_set))
                        continue

                if beta_uncorrected_col is not None:
                    try:
                        beta_uncorrected = float(cols[beta_uncorrected_col])
                        if min_gene_set_beta_uncorrected is not None and beta_uncorrected < min_gene_set_beta_uncorrected:
                            continue

                    except ValueError:
                        if not cols[beta_uncorrected_col] == "NA":
                            warn("Skipping unconvertible beta_uncorrected value %s for gene_set %s" % (cols[beta_uncorrected_col], gene_set))
                        continue

                if pheno in pheno_to_ind:
                    pheno_ind = pheno_to_ind[pheno]
                else:
                    pheno_ind = len(phenos)
                    pheno_to_ind[pheno] = pheno_ind
                    phenos.append(pheno)

                gene_set_ind = None

                if self.gene_sets is not None:
                    if gene_set not in self.gene_set_to_ind:
                        ignored += 1
                        continue

                    gene_set_ind = self.gene_set_to_ind[gene_set]
                    if gene_set_ind is not None:
                        subset_mask[gene_set_ind] = True
                else:
                    #store these in all cases to be able to check for duplicate gene sets in the input
                    gene_set_to_ind[gene_set] = len(gene_sets)
                    gene_sets.append(gene_set)

                if return_only_ids:
                    read_ids.add(gene_set)
                    continue

                if gene_set_ind is not None:
                    col.append(gene_set_ind)
                    row.append(pheno_ind)

                    if beta_uncorrected is not None:
                        betas_uncorrected.append(beta_uncorrected)
                    else:
                        betas_uncorrected.append(beta)
                        
                    if beta is not None:
                        betas.append(beta)
                    else:
                        betas.append(beta_uncorrected)

            log("Done reading --stats-in-file", DEBUG)

        if return_only_ids:
            return read_ids

        if update_X:
            if self.gene_sets is not None:
                log("Subsetting matrices", DEBUG)
                #need to subset existing matrices
                if sum(subset_mask) != len(subset_mask):
                    warn("Excluding %s values from previously loaded files because absent from --stats-in file" % (len(subset_mask) - sum(subset_mask)))
                    self.subset_gene_sets(subset_mask, keep_missing=True)
                log("Done subsetting matrices", DEBUG)

            self._set_X(self.X_orig, self.genes, self.gene_sets, skip_N=True)

        #store the phenotypes
        if self.phenos is not None:
            bail("Bug in code: cannot call this function if phenos have already been read")
            
        self.phenos = phenos

        self.pheno_to_ind = self._construct_map_to_ind(phenos)

        #uniquify if needed
        betas = np.array(betas)
        betas_uncorrected = np.array(betas_uncorrected)
        row = np.array(row)
        col = np.array(col)
        indices = np.array(list(zip(row, col)))
        _, unique_indices = np.unique(indices, axis=0, return_index=True)
        if len(unique_indices) < len(row):
            warn("Found %d duplicate values; ignoring duplicates" % (len(row) - len(unique_indices)))

        betas = betas[unique_indices]
        betas_uncorrected = betas_uncorrected[unique_indices]
        row = row[unique_indices]
        col = col[unique_indices]

        self.X_phewas_beta = sparse.csc_matrix((betas, (row, col)), shape=(len(self.phenos), len(self.gene_sets)))
        self.X_phewas_beta_uncorrected = sparse.csc_matrix((betas_uncorrected, (row, col)), shape=(len(self.phenos), len(self.gene_sets)))


    def read_gene_phewas_bfs(self, gene_phewas_bfs_in, gene_phewas_bfs_id_col=None, gene_phewas_bfs_pheno_col=None, anchor_genes=None, anchor_phenos=None, gene_phewas_bfs_log_bf_col=None, gene_phewas_bfs_combined_col=None, gene_phewas_bfs_prior_col=None, min_value=None, **kwargs):

        #require X matrix

        if gene_phewas_bfs_in is None:
            bail("Require --gene-bfs-in or --gene-phewas-bfs-in for this operation")

        log("Reading --gene-phewas-bfs-in file %s" % gene_phewas_bfs_in, INFO)

        if self.genes is None:
            bail("Need to initialixe --X before reading gene_phewas")

        Ys = None
        combineds = None
        priors = None

        row = []
        col = []
        with open_gz(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
            header_cols = gene_phewas_bfs_fh.readline().strip().split()
            if gene_phewas_bfs_id_col is None:
                gene_phewas_bfs_id_col = "Gene"
            if gene_phewas_bfs_pheno_col is None:
                gene_phewas_bfs_pheno_col = "Pheno"

            id_col = self._get_col(gene_phewas_bfs_id_col, header_cols)

            pheno_col = self._get_col(gene_phewas_bfs_pheno_col, header_cols)

            bf_col = None
            if gene_phewas_bfs_log_bf_col is not None:
                bf_col = self._get_col(gene_phewas_bfs_log_bf_col, header_cols)
            else:
                bf_col = self._get_col("log_bf", header_cols, False)

            combined_col = None
            if gene_phewas_bfs_combined_col is not None:
                combined_col = self._get_col(gene_phewas_bfs_combined_col, header_cols, True)
            else:
                combined_col = self._get_col("combined", header_cols, False)

            prior_col = None
            if gene_phewas_bfs_prior_col is not None:
                prior_col = self._get_col(gene_phewas_bfs_prior_col, header_cols, True)
            else:
                prior_col = self._get_col("prior", header_cols, False)

            if bf_col is not None:
                Ys  = []
            if combined_col is not None:
                combineds = []
            if prior_col is not None:
                priors = []

            if self.phenos is not None:
                phenos = copy.copy(self.phenos)
                pheno_to_ind = copy.copy(self.pheno_to_ind)
            else:
                phenos = []
                pheno_to_ind = {}

            self.num_gene_phewas_filtered = 0
            for line in gene_phewas_bfs_fh:
                cols = line.strip().split()
                if id_col >= len(cols) or pheno_col >= len(cols) or (bf_col is not None and bf_col >= len(cols)) or (combined_col is not None and combined_col >= len(cols)) or (prior_col is not None and prior_col >= len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]

                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                if gene not in self.gene_to_ind:
                    continue

                pheno = cols[pheno_col]

                cur_combined = None
                if combined_col is not None:
                    try:
                        combined = float(cols[combined_col])
                    except ValueError:
                        if not cols[combined_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[combined_col], gene))
                        continue

                    if min_value is not None and combined < min_value:
                        self.num_gene_phewas_filtered += 1
                        continue

                    cur_combined = combined

                if bf_col is not None:
                    try:
                        bf = float(cols[bf_col])
                    except ValueError:
                        if not cols[bf_col] == "NA":
                            warn("Skipping unconvertible value %s for gene %s and pheno %s" % (cols[bf_col], gene, pheno))
                        continue

                    if min_value is not None and combined_col is None and bf < min_value:
                        self.num_gene_phewas_filtered += 1
                        continue

                    cur_Y = bf

                if prior_col is not None:
                    try:
                        prior = float(cols[prior_col])
                    except ValueError:
                        if not cols[prior_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[prior_col], gene))
                        continue

                    if min_value is not None and combined_col is None and bf_col is None and prior < min_value:
                        self.num_gene_phewas_filtered += 1
                        continue

                    cur_prior = prior


                if pheno not in pheno_to_ind:
                    pheno_to_ind[pheno] = len(phenos)
                    phenos.append(pheno)

                pheno_ind = pheno_to_ind[pheno]

                if combineds is not None:
                    combineds.append(cur_combined)
                if Ys is not None:
                    Ys.append(cur_Y)
                if priors is not None:
                    priors.append(cur_prior)

                col.append(pheno_ind)
                row.append(self.gene_to_ind[gene])

        #update what's stored internally
        num_added_phenos = 0
        if self.phenos is not None and len(self.phenos) < len(phenos):
            num_added_phenos = len(phenos) - len(self.phenos)

        if num_added_phenos > 0:
            if self.X_phewas_beta is not None:
                self.X_phewas_beta = sparse.csc_matrix(sparse.vstack((self.X_phewas_beta, sparse.csc_matrix((num_added_phenos, self.X_phewas_beta.shape[1])))))
            if self.X_phewas_beta_uncorrected is not None:
                self.X_phewas_beta_uncorrected = sparse.csc_matrix(sparse.vstack((self.X_phewas_beta_uncorrected, sparse.csc_matrix((num_added_phenos, self.X_phewas_beta_uncorrected.shape[1])))))

        self.phenos = phenos
        pheno_to_ind = self._construct_map_to_ind(phenos)

        #uniquify if needed
        row = np.array(row)
        col = np.array(col)
        indices = np.array(list(zip(row, col)))
        _, unique_indices = np.unique(indices, axis=0, return_index=True)
        if len(unique_indices) < len(row):
            warn("Found %d duplicate values; ignoring duplicates" % (len(row) - len(unique_indices)))

        row = row[unique_indices]
        col = col[unique_indices]

        if combineds is not None:
            combineds = np.array(combineds)[unique_indices]
            self.gene_pheno_combined_prior_Ys = sparse.csc_matrix((combineds, (row, col)), shape=(len(self.genes), len(self.phenos)))

        if Ys is not None:
            Ys = np.array(Ys)[unique_indices]
            self.gene_pheno_Y = sparse.csc_matrix((Ys, (row, col)), shape=(len(self.genes), len(self.phenos)))
            

        if priors is not None:
            priors = np.array(priors)[unique_indices]
            self.gene_pheno_priors = sparse.csc_matrix((priors, (row, col)), shape=(len(self.genes), len(self.phenos)))
        
        self.anchor_gene_mask = None
        if anchor_genes is not None:
            self.anchor_gene_mask = np.array([x in anchor_genes for x in self.genes])
            if np.sum(self.anchor_gene_mask) == 0:
                bail("Couldn't find any match for %s" % list(anchor_genes))

        self.anchor_pheno_mask = None
        if anchor_phenos is not None:
            self.anchor_pheno_mask = np.array([x in anchor_phenos for x in self.phenos])
            if np.sum(self.anchor_pheno_mask) == 0:
                bail("Couldn't find any match for %s" % list(anchor_phenos))


    def calculate_gene_set_statistics(self, gwas_in=None, exomes_in=None, positive_controls_in=None, positive_controls_list=None, gene_bfs_in=None, gene_percentiles_in=None, gene_zs_in=None, Y=None, show_progress=True, max_gene_set_p=None, run_gls=False, run_logistic=True, max_for_linear=0.95, run_corrected_ols=False, use_sampling_for_betas=None, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, skip_V=False, **kwargs):
        if self.X_orig is None:
            bail("Error: X is required")
        #now calculate the betas and p-values

        log("Calculating gene set statistics", INFO)

        if Y is None:
            Y = self.Y_for_regression

        if Y is None:
            if gwas_in is None and exomes_in is None and gene_bfs_in is None and gene_percentiles_in is None and gene_zs_in is None and positive_controls_in is None and positive_controls_list is None:
                bail("Need --gwas-in or --exomes-in or --gene-bfs-in or --gene-percentiles-in or --gene-zs-in or --positive-controls-in")

            log("Reading Y within calculate_gene_set_statistics; parameters may not be honored")
            self.read_Y(gwas_in=gwas_in, exomes_in=exomes_in, positive_controls_in=positive_controls_in, positive_controls_list=positive_controls_list, gene_bfs_in=gene_bfs_in, gene_percentiles_in=gene_percentiles_in, gene_zs_in=gene_zs_in, **kwargs)


        #FIXME: need to make this so don't always read in correlations, and add priors where needed
        #can move this inside of the Y is None loop
        #but -- if compute correlation distance function is true and gene_cor_file is none and gene_loc file is not None, then we need to redo this
        #and: if gene_cor_file is not none, then we need to update the correlation matrix to account for the priors
        #To decrease correlation, we first convert cor to covaraince (multiply by np.var(Y)) then divide by np.var(Y) + np.var(prior). For np.sd prior, we can either use a fixed value (the sd of priors across all genes) or we can use the actual y values (and thus do np.sqrt(np.var(Y) + prior1)np.sqrt(np.var(Y) + prior2).
        #and, finally: we always need to call set Y here
        if run_gls:
            run_corrected_ols = False

        if (run_gls or run_corrected_ols) and self.y_corr is None:
            correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)

            #convert X and Y to their new values
            min_correlation = 0.05
            self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, Y_corr_m=correlation_m, store_cholesky=run_gls, store_corr_sparse=run_corrected_ols, skip_V=True, skip_scale_factors=True, min_correlation=min_correlation)

        #subset gene sets to remove empty ones first
        #number of gene sets in each gene set
        col_sums = self.get_col_sums(self.X_orig, num_nonzero=True)
        self.subset_gene_sets(col_sums > 0, keep_missing=False, skip_V=True, skip_scale_factors=True)

        #CAN REMOVE
        #mean of Y is now zero
        #self.beta_tildes = self.scale_factors * ((self.X_orig.T.dot(Y_clean) / len(Y_clean)) - (self.mean_shifts * np.mean(Y_clean))) / variances
        self._set_scale_factors()

        #self.is_logistic = run_logistic

        #if the maximum Y is large, switch to logistic regression (to avoid being too strong)
        Y_to_use = self.Y_for_regression
        Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))

        if not run_logistic and np.max(Y) > max_for_linear and (use_sampling_for_betas is None or use_sampling_for_betas < 1):
            log("Switching to logistic sampling due to high Y values", DEBUG)
            run_logistic = True
            use_sampling_for_betas = 1

        if use_sampling_for_betas is not None:
            self._record_param("sampling_for_betas", use_sampling_for_betas)

        if use_sampling_for_betas is not None and use_sampling_for_betas > 0:

            #handy option in case we want to see what sampling looks like outside of gibbs
            avg_beta_tildes = np.zeros(len(self.gene_sets))
            avg_z_scores = np.zeros(len(self.gene_sets))
            tot_its = 0
            for iteration_num in range(use_sampling_for_betas):
                log("Sampling iteration %d..." % (iteration_num+1))
                p_sample_m = np.zeros(Y.shape)
                p_sample_m[np.random.random(Y.shape) < Y] = 1
                Y = p_sample_m

                (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig, Y, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

                avg_beta_tildes += beta_tildes
                avg_z_scores += z_scores
                tot_its += 1

            self.beta_tildes = avg_beta_tildes / tot_its
            self.z_scores = avg_z_scores / tot_its

            self.p_values = 2*scipy.stats.norm.cdf(-np.abs(self.z_scores))
            self.ses = np.full(self.beta_tildes.shape, 100.0)
            self.ses[self.z_scores != 0] = np.abs(self.beta_tildes[self.z_scores != 0] / self.z_scores[self.z_scores != 0])

        elif run_logistic:
                (self.beta_tildes, self.ses, self.z_scores, self.p_values, self.se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig, Y, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

        else:
            if run_gls:
                #Y has already been whitened
                #dot_product = np.array([])
                y_var = self.y_fw_var
                Y = self.Y_fw
                #OLD CODE
                #as an optimization, multiply original X by fully whitened Y, rather than half by half
                #for X_b, begin, end, batch in self._get_X_blocks():
                #    #calculate mean shifts
                #    dot_product = np.append(dot_product, X_b.T.dot(self.Y_w) / len(self.Y_w))
                #dot_product = self.X_orig.T.dot(self.Y_fw) / len(self.Y_fw)
            else:
                #Technically, we could use the above code for this case, since X_blocks will returned unwhitened matrix
                #But, probably faster to keep sparse multiplication? Might be worth revisiting later to see if there actually is a performance gain
                #We can use original X here because we know that whitening will occur only for GLS
                #assert this to be sure
                assert(not self.scale_is_for_whitened)
                Y = copy.copy(self.Y_for_regression)

                y_var = self.y_var

                #OLD CODE
                #dot_product = self.X_orig.T.dot(self.Y) / len(self.Y)

            #variances = np.power(self.scale_factors, 2)
            #multiply by scale factors because we store beta_tilde in units of scaled X
            #self.beta_tildes = self.scale_factors * np.array(dot_product) / variances
            #self.ses = self.scale_factors * np.sqrt(y_var) / (np.sqrt(variances * len(self.Y)))
            #self.z_scores = self.beta_tildes / self.ses
            #self.p_values = 2*scipy.stats.norm.cdf(-np.abs(self.z_scores))

            (self.beta_tildes, self.ses, self.z_scores, self.p_values, self.se_inflation_factors) = self._compute_beta_tildes(self.X_orig, Y, y_var, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

        if correct_betas_mean or correct_betas_var:
            (self.beta_tildes, self.ses, self.z_scores, self.p_values, self.se_inflation_factors) = self._correct_beta_tildes(self.beta_tildes, self.ses, self.se_inflation_factors, self.total_qc_metrics, self.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, fit=False)

        self.X_orig_missing_gene_sets = None
        self.mean_shifts_missing = None
        self.scale_factors_missing = None
        self.is_dense_gene_set_missing = None
        self.ps_missing = None
        self.sigma2s_missing = None

        self.beta_tildes_missing = None
        self.p_values_missing = None
        self.ses_missing = None
        self.z_scores_missing = None

        self.total_qc_metrics_missing = None
        self.mean_qc_metrics_missing = None

        if max_gene_set_p is not None:
            gene_set_mask = self.p_values <= max_gene_set_p
            if np.sum(gene_set_mask) == 0 and len(self.p_values) > 0:
                gene_set_mask = self.p_values == np.min(self.p_values)
            log("Keeping %d gene sets that passed threshold of p<%.3g" % (np.sum(gene_set_mask), max_gene_set_p))
            self.subset_gene_sets(gene_set_mask, keep_missing=True, skip_V=True)

            if len(self.gene_sets) < 1:
                log("No gene sets left!")
                return

        #self.max_gene_set_p = max_gene_set_p

    #FIXME: Update calls to use includes_non_missing and from_osc
    def set_p(self, p):
        #if p is None:
        #    log("Set p called with p=%s" % p, TRACE)
        #else:
        #    log("Set p called with p=%.3g" % p, TRACE)

        if p is not None:
            if p > 1:
                p = 1
            if p < 0:
                p = 0
        self.p = p

    def get_sigma2(self, convert_sigma_to_external_units=False):
        if self.sigma2 is not None and convert_sigma_to_external_units and self.sigma_power is not None:
            if self.scale_factors is not None:
                if self.is_dense_gene_set is not None and np.sum(~self.is_dense_gene_set) > 0:
                    return self.sigma2 * np.mean(np.power(self.scale_factors[~self.is_dense_gene_set], self.sigma_power - 2))
                else:
                    return self.sigma2 * np.mean(np.power(self.scale_factors, self.sigma_power - 2))
            else:
                return self.sigma2 * np.power(self.MEAN_MOUSE_SCALE, self.sigma_power - 2)

        else:
            return self.sigma2

    def get_scaled_sigma2(self, scale_factors, sigma2, sigma_power, sigma_threshold_k=None, sigma_threshold_xo=None):
        threshold = 1
        if sigma_threshold_k is not None and sigma_threshold_xo is not None:
            threshold =  1 / (1 + np.exp(-sigma_threshold_k * (scale_factors - sigma_threshold_xo)))

        zero_mask = None
        if len(scale_factors.shape) == 0:
            if scale_factors == 0:
                return 0
        else:
            zero_mask = scale_factors == 0
            scale_factors[zero_mask] = 1

        result = threshold * sigma2 * np.power(scale_factors, sigma_power)
        if zero_mask is not None:
            result[zero_mask] = 0

        return result

    def set_sigma(self, sigma2, sigma_power, sigma2_osc=None, sigma2_se=None, sigma2_p=None, sigma2_scale_factors=None, convert_sigma_to_internal_units=False):

        #if sigma2 is None:
        #    log("Set sigma called with sigma2=%s" % sigma2, TRACE)
        #else:
        #    log("Set sigma called with sigma2=%.3g" % sigma2, TRACE)            

        #WARNING: sigma storage is not handled optimally right now
        #if self.sigma_power=None, then sigma2 is in units of internal beta (because it is constant internally)
        #so, sigma2 / np.square(self.scale_factors[i]) is the external beta unit
        #if self.sigma_power != None, then sigma2 is in units of external beta (because it is constant externally)
        #so, sigma2 * np.power(self.scale_factors[i], self.sigma_power) is the internal beta unit
        #This setter does not validate any of this, so
        #1. when getting sigma, you must convert to internal sigma units
        #2. when setting sigma, you must pass in external units if const sigma and internal units if not


        self.sigma_power = sigma_power
        if sigma_power is None:
            #default is to have constant sigma in external units of beta, so beta is 2
            sigma_power = 2

        if convert_sigma_to_internal_units:
            #we divide by expected value of scale ** (power - 2) because:
            #beta_internal_j = beta_j * scale_factors_j -> beta_j = beta_internal_j / scale_factors_j
            #beta_internal_j ~ N(0, sigma2 * scale_factors_j ** sigma_power) -> beta_j ~ N(0, sigma2 * scale_factors_j ** (sigma_power-2))
            #sigma2_ext = E[sigma2 * scale_factors_j ** (sigma_power - 2)] = sigma2 * E[scale_factors_j ** (sigma_power - 2)]
            #sigma2 = sigma2_ext / E[scale_factors_j ** (sigma_power - 2)]
            if self.scale_factors is not None:
                if self.is_dense_gene_set is not None and np.sum(~self.is_dense_gene_set) > 0:
                    self.sigma2 = sigma2 / np.mean(np.power(self.scale_factors[~self.is_dense_gene_set], self.sigma_power - 2))
                else:
                    self.sigma2 = sigma2 / np.mean(np.power(self.scale_factors, self.sigma_power - 2))
            else:
                self.sigma2 = sigma2 / np.power(self.MEAN_MOUSE_SCALE, self.sigma_power - 2)
        else:
            self.sigma2 = sigma2

        if sigma2_osc is not None:
            self.sigma2_osc = sigma2_osc

        if sigma2_scale_factors is None:
            sigma2_scale_factors = self.scale_factors

        if sigma2_se is not None:
            self.sigma2_se = sigma2_se
        if self.sigma2_p is not None:
            self.sigma2_p = sigma2_p

        if self.sigma2 is None and self.sigma2_osc is None:
            return

        sigma2_for_var = self.sigma2_osc if self.sigma2_osc is not None else self.sigma2

        if sigma2_for_var is not None and sigma2_scale_factors is not None:
            if self.sigma_power is None:
                self.sigma2_total_var = sigma2_for_var * len(sigma2_scale_factors)
            else:
                self.sigma2_total_var = sigma2_for_var * np.sum(np.square(sigma2_scale_factors))

        if self.sigma2_total_var is not None and self.sigma2_se is not None:
            self.sigma2_total_var_lower = self.sigma2_total_var * (sigma2_for_var - 1.96 * self.sigma2_se)/(sigma2_for_var)
            self.sigma2_total_var_upper = self.sigma2_total_var * (sigma2_for_var + 1.96 * self.sigma2_se)/(sigma2_for_var)

        #minimum bound
        if self.sigma2 is None:
            return

    def calculate_sigma(self, V, sigma_power=None, chisq_threshold=None, chisq_dynamic=False, desired_intercept_difference=1.3):
        if self.z_scores is None:
            bail("Cannot calculate sigma with no stats loaded!")
        if V is None:
            V = self._get_V()
        if len(self.z_scores) == 0:
            bail("No gene sets were in both V and stats!")
        self.sigma_power = sigma_power
       
        log("Calculating OSC", DEBUG)

        #generating batches are most expensive part here, so do each one only once
        self.osc = np.zeros(self.X_orig.shape[1])
        if self.X_orig_missing_gene_sets is not None:
            self.osc_missing = np.zeros(self.X_orig_missing_gene_sets.shape[1])

        for X_b1, begin1, end1, batch1 in self._get_X_blocks(whiten=False, full_whiten=True):
            #begin block for calculating OSC between non-missing gene sets and non-missing gene sets
            for X_b2, begin2, end2, batch2 in self._get_X_blocks(start_batch=batch1, whiten=False, full_whiten=False):
                self.osc[begin1:end1] = np.add(self.osc[begin1:end1], np.sum(np.power(self._compute_V(X_b1, self.mean_shifts[begin1:end1], self.scale_factors[begin1:end1], X_orig2=X_b2, mean_shifts2=self.mean_shifts[begin2:end2], scale_factors2=self.scale_factors[begin2:end2]), 2), axis=1))
                if not batch1 == batch2:
                    self.osc[begin2:end2] = np.add(self.osc[begin2:end2], np.sum(np.power(self._compute_V(X_b2, self.mean_shifts[begin2:end2], self.scale_factors[begin2:end2], X_orig2=X_b1, mean_shifts2=self.mean_shifts[begin1:end1], scale_factors2=self.scale_factors[begin1:end1]), 2), axis=1))
            #end block for calculating OSC between non-missing gene sets and non-missing gene sets

            if self.X_orig_missing_gene_sets is not None:
                for X_m_b1, m_begin1, m_end1, m_batch1 in self._get_X_blocks(get_missing=True, whiten=False, full_whiten=False):
                    #since we have the missing blocks, do the missing/missing osc as well
                    #but only want to do this once
                    #osc between non-missing and missing
                    self.osc[begin1:end1] = np.add(self.osc[begin1:end1], np.sum(np.power(self._compute_V(X_b1, self.mean_shifts[begin1:end1], self.scale_factors[begin1:end1], X_orig2=X_m_b1, mean_shifts2=self.mean_shifts_missing[m_begin1:m_end1], scale_factors2=self.scale_factors_missing[m_begin1:m_end1]), 2), axis=1))
                    #osc between missing and non-missing
                    self.osc_missing[m_begin1:m_end1] = np.add(self.osc_missing[m_begin1:m_end1], np.sum(np.power(self._compute_V(X_m_b1, self.mean_shifts_missing[m_begin1:m_end1], self.scale_factors_missing[m_begin1:m_end1], X_orig2=X_b1, mean_shifts2 = self.mean_shifts[begin1:end1], scale_factors2 = self.scale_factors[begin1:end1]), 2), axis=1))

        if self.X_orig_missing_gene_sets is not None:
            for X_m_b1, m_begin1, m_end1, m_batch1 in self._get_X_blocks(get_missing=True, whiten=False, full_whiten=True):
                for X_m_b2, m_begin2, m_end2, m_batch2 in self._get_X_blocks(get_missing=True, whiten=False, full_whiten=False, start_batch=m_batch1):
                    self.osc_missing[m_begin1:m_end1] = np.add(self.osc_missing[m_begin1:m_end1], np.sum(np.power(self._compute_V(X_m_b1, self.mean_shifts_missing[m_begin1:m_end1], self.scale_factors_missing[m_begin1:m_end1], X_orig2=X_m_b2, mean_shifts2=self.mean_shifts_missing[m_begin2:m_end2], scale_factors2=self.scale_factors_missing[m_begin2:m_end2]), 2), axis=1))
                    if not m_batch1 == m_batch2:
                        self.osc_missing[m_begin2:m_end2] = np.add(self.osc_missing[m_begin2:m_end2], np.sum(np.power(self._compute_V(X_m_b2, self.mean_shifts_missing[m_begin2:m_end2], self.scale_factors_missing[m_begin2:m_end2], X_orig2=X_m_b1, mean_shifts2=self.mean_shifts_missing[m_begin1:m_end1], scale_factors2=self.scale_factors_missing[m_begin1:m_end1]), 2), axis=1))

        #X_osc is in units of standardized X
        self.X_osc = self.osc/np.square(self.ses)
        if self.X_orig_missing_gene_sets is not None:
            self.X_osc_missing = self.osc_missing/np.square(self.ses_missing)
            osc = np.append(self.osc, self.osc_missing)
            denominator = np.square(np.append(self.ses, self.ses_missing))
            scale_factors = np.append(self.scale_factors, self.scale_factors_missing)
            Y_chisq=np.square(np.append(self.z_scores, self.z_scores_missing))
        else:
            self.osc_missing = None
            self.X_osc_missing = None
            osc = self.osc
            denominator = np.square(self.ses)
            scale_factors = self.scale_factors
            Y_chisq=np.square(self.z_scores)

        X_osc = osc/denominator
        if self.sigma_power is not None:
            #all of the X_osc have been scaled by 1/scale_factor**2 (because each X is scaled by 1/scale_factor)
            #so we need to multiply by scale_factor
            if np.sum(~self.is_dense_gene_set) > 0:
                X_osc[~self.is_dense_gene_set] = X_osc[~self.is_dense_gene_set] * np.power(scale_factors[~self.is_dense_gene_set], self.sigma_power)
                X_osc[self.is_dense_gene_set] = X_osc[self.is_dense_gene_set] * np.power(np.mean(scale_factors[~self.is_dense_gene_set]), self.sigma_power)
                self.X_osc[~self.is_dense_gene_set] = self.X_osc[~self.is_dense_gene_set] * np.power(self.scale_factors[~self.is_dense_gene_set], self.sigma_power)
                self.X_osc[self.is_dense_gene_set] = self.X_osc[self.is_dense_gene_set] * np.power(np.mean(self.scale_factors[~self.is_dense_gene_set]), self.sigma_power)

                if self.X_orig_missing_gene_sets is not None:
                    self.X_osc_missing[~self.is_dense_gene_set_missing] = self.X_osc_missing[~self.is_dense_gene_set_missing] * np.power(self.scale_factors_missing[~self.is_dense_gene_set_missing], self.sigma_power)
                    self.X_osc_missing[self.is_dense_gene_set_missing] = self.X_osc_missing[self.is_dense_gene_set_missing] * np.power(np.mean(self.scale_factors[~self.is_dense_gene_set]), self.sigma_power)
            else:
                X_osc[self.is_dense_gene_set] = X_osc[self.is_dense_gene_set] * np.power(np.mean(scale_factors), self.sigma_power)
                self.X_osc[self.is_dense_gene_set] = self.X_osc[self.is_dense_gene_set] * np.power(np.mean(self.scale_factors), self.sigma_power)
                if self.X_orig_missing_gene_sets is not None:
                    self.X_osc_missing[self.is_dense_gene_set_missing] = self.X_osc_missing[self.is_dense_gene_set_missing] * np.power(np.mean(self.scale_factors), self.sigma_power)

        #X_osc is N l_j
        #osc is l_j

        #OLD
        #osc_weights = 1/(np.square(1 + X_osc) * X_osc)
        #NEW
        tau = (np.power(np.mean(Y_chisq), 2) - 1) / (np.mean(X_osc))
        osc_weights = 1/(osc * np.square(1 + X_osc * tau))

        #OLD
        #self.osc_weights = 1/(np.square(1 + self.X_osc) * self.X_osc)
        #NEW
        self.osc_weights = 1/(self.osc * np.square(1 + tau * self.X_osc))

        if self.X_orig_missing_gene_sets is not None:
            #OLD
            #self.osc_weights_missing = 1/(np.square(1 + self.X_osc_missing) * self.X_osc_missing)
            #NEW
            self.osc_weights_missing = 1/(self.osc_missing * np.square(1 + tau * self.X_osc_missing))
        else:
            self.osc_weights_missing = None

        orig_mask = ~(np.isnan(Y_chisq) | np.isinf(Y_chisq) | np.isnan(X_osc) | np.isinf(X_osc) | np.isnan(osc_weights) | np.isinf(osc_weights))

        if chisq_dynamic:
            cur_chisq_threshold = max(Y_chisq)
            best_intercept_difference = None
            best_chisq_threshold = None
            min_chisq_threshold = 5
        elif chisq_threshold is not None:
            cur_chisq_threshold = chisq_threshold
        else:
            cur_chisq_threshold = np.inf

        log("Running OSC regressions", DEBUG)

        while True:

            mask = np.logical_and(orig_mask, Y_chisq < cur_chisq_threshold)

            #run to get intercept
            constant_term = np.ones(len(X_osc[mask]))
            X = np.vstack((X_osc[mask], constant_term))
            Y = Y_chisq[mask] 
            #add weights for WLS
            X[0,] = X[0,] * np.sqrt(osc_weights[mask])
            X[1,] = X[1,] * np.sqrt(osc_weights[mask])
            Y = Y * np.sqrt(osc_weights[mask])
            try:
                mat_inv = np.linalg.inv(X.dot(X.T) + np.eye)
            except np.linalg.LinAlgError:
                mat_inv = np.linalg.inv(X.dot(X.T) + 0.2 * np.eye(X.shape[0]))

            result = mat_inv.dot(X.dot(Y))
            result_ses = np.sqrt(np.diag(np.var(Y - result.dot(X)) * mat_inv))
            cur_beta = result[0]
            cur_beta_se = result_ses[0]
            cur_beta_z = cur_beta / cur_beta_se
            cur_beta_p = 2*scipy.stats.norm.cdf(-np.abs(cur_beta_z))
            cur_intercept = result[1]
            cur_intercept_se = result_ses[1]
            cur_intercept_z = cur_intercept / cur_intercept_se
            cur_intercept_p = 2*scipy.stats.norm.cdf(-np.abs(cur_intercept_z))

            def __write_results(beta, beta_se, beta_p, intercept=None, intercept_se=None, intercept_p=None, level=INFO):
                log("=================================================", level=level)
                log("value        coef%s      std err      P" % ("" if beta > 0 and (intercept is None or intercept > 0) else " "), level=level)
                log("-------------------------------------------------", level=level)
                log("beta         %.4g%s    %.4g       %.4g" % (beta, " " if beta > 0 else "", beta_se, beta_p), level=level)
                if intercept is not None:
                    log("intercept    %.4g%s    %.4g       %.3g" % (intercept, " " if intercept > 0 else "", intercept_se, intercept_p), level=level)
                log("================================================", level=level)

            log("Results from full regression (chisq-threshold=%.3g):" % cur_chisq_threshold, TRACE)
            __write_results(cur_beta, cur_beta_se, cur_beta_p, cur_intercept, cur_intercept_se, cur_intercept_p, TRACE)

            #log("Results from regression with pinned intercept at 1:")

            #X = X[0,:]
            #Y = (Y_chisq[mask] - 1) * np.sqrt(osc_weights[mask])
            #beta = X.dot(Y) / X.dot(X)
            #beta_se = np.sqrt(np.var(Y - beta * X) / X.dot(X))
            #beta_z = beta / beta_se
            #beta_p = 2*scipy.stats.norm.cdf(-np.abs(beta_z))
            #__write_results(beta, beta_se, beta_p)

            #first log the results with the intercept
            if not chisq_dynamic:
                best_chisq_threshold = cur_chisq_threshold
                intercept = cur_intercept
                beta = cur_beta
                beta_se = cur_beta_se
                beta_p = cur_beta_p
                break
            else:
                if best_intercept_difference is None or np.abs(cur_intercept - 1) < best_intercept_difference:
                    best_intercept_difference = np.abs(cur_intercept - 1)
                    best_chisq_threshold=cur_chisq_threshold
                    intercept = cur_intercept
                    beta = cur_beta
                    beta_se = cur_beta_se
                    beta_p = cur_beta_p
                if cur_chisq_threshold < min_chisq_threshold or best_intercept_difference < desired_intercept_difference:
                    break
                else:
                    cur_chisq_threshold /= 1.5

        log("Results from full regression (chisq-threshold=%.3g):" % best_chisq_threshold, INFO)
        __write_results(cur_beta, cur_beta_se, cur_beta_p, cur_intercept, cur_intercept_se, cur_intercept_p, INFO)
        log("Final sigma results:")
        self.intercept = intercept
        self.set_sigma(beta, self.sigma_power, sigma2_osc=beta, sigma2_se=beta_se, sigma2_p=beta_p, sigma2_scale_factors=scale_factors)
        #self.write_params(None)

        #now log the results with the pinned
        #log("Results with pinned intercept:")
        #self.set_sigma(beta, self.sigma_power, beta_se, beta_p, sigma2_scale_factors=scale_factors)
        #self.write_sigma(None)

    def write_params(self, output_file):
        if output_file is not None:
            log("Writing params to %s" % output_file, INFO)
            params_fh = open(output_file, 'w')

            params_fh.write("Parameter\tVersion\tValue\n")
            for param in self.param_keys:
                if type(self.params[param]) == list:
                    values = self.params[param]
                else:
                    values = [self.params[param]]
                for i in range(len(values)):
                    params_fh.write("%s\t%s\t%s\n" % (param, i + 1, values[i]))
                        
            params_fh.close()

    def read_betas(self, betas_in):

        betas_format = "<gene_set_id> <beta>"

        if self.betas_in is None:
            bail("Operation requires --beta-in\nformat: %s" % (self.betas_format))

        log("Reading --betas-in file %s" % self.betas_in, INFO)

        with open_gz(betas_in) as betas_fh:
            id_col = 0
            beta_col = 1

            if self.gene_sets is not None:
                self.betas = np.zeros(len(self.gene_sets))
                subset_mask = np.array([False] * len(self.gene_sets))
            else:
                self.betas = []

            gene_sets = []
            gene_set_to_ind = {}

            ignored = 0
            for line in betas_fh:
                cols = line.strip().split()
                if id_col > len(cols) or beta_col > len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene_set = cols[id_col]
                if gene_set in gene_set_to_ind:
                    warn("Already seen gene set %s; only considering first instance" % (gene_set))
                try:
                    beta = float(cols[beta_col])
                except ValueError:
                    if not cols[beta_col] == "NA":
                        warn("Skipping unconvertible beta value %s for gene_set %s" % (cols[beta_col], gene_set))
                    continue
                
                gene_set_ind = None
                if self.gene_sets is not None:
                    if gene_set not in self.gene_set_to_ind:
                        ignored += 1
                        continue
                    gene_set_ind = self.gene_set_to_ind[gene_set]
                    if gene_set_ind is not None:
                        self.betas[gene_set_ind] = beta
                        subset_mask[gene_set_ind] = True
                else:
                    self.betas.append(beta)
                    #store these in all cases to be able to check for duplicate gene sets in the input
                    gene_set_to_ind[gene_set] = len(gene_sets)
                    gene_sets.append(gene_set)

            if self.gene_sets is not None:
                #need to subset existing marices
                if ignored > 0:
                    warn("Ignored %s values from --betas-in file because absent from previously loaded files" % ignored)
                if sum(subset_mask) != len(subset_mask):
                    warn("Excluding %s values from previously loaded files because absent from --betas-in file" % (len(subset_mask) - sum(subset_mask)))
                    self.subset_gene_sets(subset_mask, keep_missing=False)
            else:
                self.gene_sets = gene_sets
                self.gene_set_to_ind = gene_set_to_ind
                self.betas = np.array(self.betas).flatten()

            if self.normalize_betas:
                self.betas -= np.mean(self.betas)

    def calculate_inf_betas(self, update_hyper_sigma=True, max_num_iter=20, eps=0.01):
        #catch the "death spiral"
        orig_sigma2 = self.sigma2
        orig_inf_betas = None
        significant_decrease = 0
        total = 0
        converged = False

        V = self._get_V()

        if self.y_corr_sparse is not None:
            V_cor = self._calculate_V_internal(self.X_orig, None, self.mean_shifts, self.scale_factors, y_corr_sparse=self.y_corr_sparse)
            V_inv = self._invert_sym_matrix(V)
        else:
            V_cor = None
            V_inv = None

        for i in range(max_num_iter):
            inf_betas = self._calculate_inf_betas(V_cor=V_cor, V_inv=V_inv, se_inflation_factors=self.se_inflation_factors)

            if not update_hyper_sigma:
                break

            if orig_inf_betas is None:
                orig_inf_betas = inf_betas

            h2 = inf_betas.dot(V).dot(inf_betas)
            if self.sigma_power is not None:
                #np.sum(sigma2 * np.square(self.scale_factors)) = h2
                new_sigma2 = h2 / np.sum(np.power(self.scale_factors, self.sigma_power))
            else:
                new_sigma2 = h2 / len(inf_betas)
            if abs(new_sigma2 - self.sigma2) / self.sigma2 < eps:
                converged = True
                break
            log("Updating sigma to: %.4g" % new_sigma2, TRACE)

            total += 1
            if new_sigma2 < self.sigma2:
                significant_decrease += 1
            self.set_sigma(new_sigma2, self.sigma_power)
            if new_sigma2 == 0:
                break

        #don't degrade it too much
        if total > 0 and not converged and float(significant_decrease) / float(total) == 1:
            log("Reverting to original sigma=%.4g due to convergence to 0" % orig_sigma2, TRACE)
            inf_betas = orig_inf_betas
            self.set_sigma(orig_sigma2, self.sigma_power)

        if self.betas is None or self.betas is self.inf_betas:
            self.betas = inf_betas

        self.inf_betas = inf_betas

        if self.gene_sets_missing is not None:
            self.betas_missing = np.zeros(len(self.gene_sets_missing))
            self.betas_uncorrected_missing = np.zeros(len(self.gene_sets_missing))
            self.inf_betas_missing = np.zeros(len(self.gene_sets_missing))

    def run_cross_val(self, cross_val_num_explore_each_direction, folds=4, cross_val_max_num_tries=2, p=None, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=4, run_logistic=True, max_for_linear=0.95, run_corrected_ols=False, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, sparse_solution=False, sparse_frac_betas=None, **kwargs):

        log("Running cross validation", DEBUG)

        if self.sigma2s is not None:
            candidate_sigma2s = self.sigma2s
        elif self.sigma2 is not None:
            candidate_sigma2s = np.array(self.sigma2).reshape((1,))
        else:
           bail("Need to have sigma set before running cross validation")

        if p is None:
           bail("Need to have p set before running cross validation")
        if self.X_orig is None:
           bail("Need to have X_orig set before running cross validation")

        Y_to_use = self.Y_for_regression
        if Y_to_use is None:
            Y_to_use = self.Y

        if Y_to_use is None:
           bail("Need to have Y set before running cross validation")

        
        D = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))
        if not run_logistic and np.max(D) > max_for_linear:
            log("Switching to logistic sampling due to high Y values", DEBUG)
            run_logistic = True

        beta_tildes_cv = np.zeros((folds, len(self.gene_sets)))
        alpha_tildes_cv = np.zeros((folds, len(self.gene_sets)))
        ses_cv = np.zeros((folds, len(self.gene_sets)))
        cv_val_masks = np.full((folds, len(Y_to_use)), False)
        for fold in range(folds):
            cv_mask = np.arange(len(Y_to_use)) % folds != fold
            cv_val_masks[fold,:] = ~cv_mask
            X_to_use = self.X_orig[cv_mask,:]
            if run_logistic:
                Y_cv = D[cv_mask]
                (beta_tildes_cv[fold,:], ses_cv[fold,:], _, _, _, alpha_tildes_cv[fold,:], _) = self._compute_logistic_beta_tildes(X_to_use, Y_cv, resid_correlation_matrix=self.y_corr_sparse[cv_mask,:][:,cv_mask])
            else:
                Y_cv = Y_to_use[cv_mask]
                (beta_tildes_cv[fold,:], ses_cv[fold,:], _, _, _) = self._compute_beta_tildes(X_to_use, Y_cv, resid_correlation_matrix=self.y_corr_sparse[cv_mask,:][:,cv_mask])

        #one parallel per sigma value to test
        cross_val_num_explore = cross_val_num_explore_each_direction * 2 + 1
        #for each parallel, need to do it with the different set of Y values
        cross_val_num_explore_with_fold = cross_val_num_explore * folds

        candidate_sigma2s_m = np.tile(candidate_sigma2s, cross_val_num_explore).reshape(cross_val_num_explore, candidate_sigma2s.shape[0])
        candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(-cross_val_num_explore_each_direction,cross_val_num_explore_each_direction+1))).T
        orig_index = cross_val_num_explore_each_direction

        for try_num in range(cross_val_max_num_tries):

            log("Sigmas to try: %s" % np.mean(candidate_sigma2s_m, axis=1), TRACE)

            #order of parallel is first by explore and then by fold

            #repeat the candidates for each fold
            candidate_sigma2s_m = np.tile(candidate_sigma2s_m, (folds, 1))

            beta_tildes_m = np.repeat(beta_tildes_cv, cross_val_num_explore, axis=0)
            ses_m = np.repeat(ses_cv, cross_val_num_explore, axis=0)
            scale_factors_m = np.tile(self.scale_factors, cross_val_num_explore_with_fold).reshape(cross_val_num_explore_with_fold, len(self.scale_factors))
            mean_shifts_m = np.tile(self.mean_shifts, cross_val_num_explore_with_fold).reshape(cross_val_num_explore_with_fold, len(self.mean_shifts))

            (betas_m, postp_m) = self._calculate_non_inf_betas(initial_p=self.p, beta_tildes=beta_tildes_m, ses=ses_m, scale_factors=scale_factors_m, mean_shifts=mean_shifts_m, sigma2s=candidate_sigma2s_m, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, V=self._get_V(), **kwargs)

            rss = np.zeros(cross_val_num_explore)
            num_Y = 0
            #different values for logistic and linear
            Y_val = Y_to_use - np.mean(Y_to_use)

            for fold in range(folds):
                #result is parallel x genes
                output_cv_mask = np.floor(np.arange(betas_m.shape[0]) / cross_val_num_explore) == fold
                cur_pred = self.X_orig[cv_val_masks[fold,:],:].dot((betas_m[output_cv_mask,:] / self.scale_factors).T).T
                rss += np.sum(np.square(cur_pred - Y_val[cv_val_masks[fold,:]]), axis=1)
                num_Y += np.sum(cv_val_masks[fold,:])

            rss /= num_Y
            best_result = np.argmin(rss)
            best_sigma2s = candidate_sigma2s_m[best_result,:]
            log("Got RSS values: %s" % (rss), TRACE)
            log("Best sigma is %.3g" % np.mean(best_sigma2s))
            log("Updating sigma from %.3g to %.3g" % (self.sigma2, np.mean(best_sigma2s)))
            if self.sigma2s is not None:
                self.sigma2s = best_sigma2s
                self.set_sigma(np.mean(best_sigma2s), self.sigma_power)
            else:
                assert(len(best_sigma2s.shape) == 1 and best_sigma2s.shape[0] == 1)
                self.set_sigma(best_sigma2s[0], self.sigma_power)

            if try_num + 1 < cross_val_max_num_tries and (best_result == 0 or best_result == (len(rss) - 1)) and best_result != orig_index:
                log("Expanding search further since best cross validation result was at boundary of search space", DEBUG)
                assert(self.sigma2s is not None or self.sigma2 is not None)
                if self.sigma2s is not None:
                    candidate_sigma2s = self.sigma2s
                else: 
                    candidate_sigma2s = np.array(self.sigma2).reshape((1,))
                candidate_sigma2s_m = np.tile(candidate_sigma2s, cross_val_num_explore).reshape(cross_val_num_explore, candidate_sigma2s.shape[0])
                if best_result == 0:
                    #extend lower
                    candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(-cross_val_num_explore+1,1))).T
                    orig_index = cross_val_num_explore - 1
                else:
                    #extend higher
                    candidate_sigma2s_m = (candidate_sigma2s_m.T * np.power(10.0, np.arange(cross_val_num_explore))).T
                    orig_index = 0
            else:
                break
        
    def calculate_non_inf_betas(self, p, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, sparse_solution=False, pre_filter_batch_size=None, pre_filter_small_batch_size=500, sparse_frac_betas=None, betas_trace_out=None, **kwargs):

        log("Calculating betas")
        (avg_betas_uncorrected_v, avg_postp_uncorrected_v) = self._calculate_non_inf_betas(p, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, assume_independent=True, V=None, **kwargs)

        avg_betas_v = np.zeros(len(self.gene_sets))
        avg_postp_v = np.zeros(len(self.gene_sets))

        initial_run_mask = avg_betas_uncorrected_v != 0
        run_mask = copy.copy(initial_run_mask)

        if pre_filter_batch_size is not None and np.sum(initial_run_mask) > pre_filter_batch_size:
            self._record_param("pre_filter_batch_size_orig", pre_filter_batch_size)

            num_batches = self._get_num_X_blocks(self.X_orig[:,initial_run_mask], batch_size=pre_filter_small_batch_size)
            if num_batches > 1:
                #try to run with small batches to see if we can zero out more
                gene_set_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], scale_factors=self.scale_factors[initial_run_mask], find_correlated_instead=pre_filter_small_batch_size)
                if len(gene_set_masks) > 0:
                    if np.sum(gene_set_masks[-1]) == 1 and len(gene_set_masks) > 1:
                        #merge singletons at the end into the one before
                        gene_set_masks[-2][gene_set_masks[-1]] = True
                        gene_set_masks = gene_set_masks[:-1]
                    if np.sum(gene_set_masks[0]) > 1:
                        V_data = []
                        V_rows = []
                        V_cols = []
                        for gene_set_mask in gene_set_masks:
                            V_block = self._calculate_V_internal(self.X_orig[:,initial_run_mask][:,gene_set_mask], self.y_corr_cholesky, self.mean_shifts[initial_run_mask][gene_set_mask], self.scale_factors[initial_run_mask][gene_set_mask])
                            orig_indices = np.where(gene_set_mask)[0]
                            V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
                            V_cols += list(np.tile(orig_indices, V_block.shape[0]))
                            V_data += list(V_block.ravel())
                            
                        V_sparse = sparse.csc_matrix((V_data, (V_rows, V_cols)), shape=(np.sum(initial_run_mask), np.sum(initial_run_mask)))

                        log("Running %d blocks to check for zeros..." % len(gene_set_masks), DEBUG)
                        (avg_betas_half_corrected_v, avg_postp_half_corrected_v) = self._calculate_non_inf_betas(p, V=V_sparse, X_orig=None, scale_factors=self.scale_factors[initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], is_dense_gene_set=self.is_dense_gene_set[initial_run_mask], ps=self.ps[initial_run_mask], sigma2s=self.sigma2s[initial_run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, **kwargs)

                        add_zero_mask = avg_betas_half_corrected_v == 0

                        if np.any(add_zero_mask):
                            #need to convert these to the original gene sets
                            map_to_full = np.where(initial_run_mask)[0]
                            #get rows and then columns in subsetted
                            set_to_zero_full = np.where(add_zero_mask)
                            #map columns in subsetted to original
                            set_to_zero_full = map_to_full[set_to_zero_full]
                            orig_zero = np.sum(run_mask)
                            run_mask[set_to_zero_full] = False
                            new_zero = np.sum(run_mask)
                            log("Found %d additional zero gene sets" % (orig_zero - new_zero),DEBUG)

        if np.sum(~run_mask) > 0:
            log("Set additional %d gene sets to zero based on uncorrected betas" % np.sum(~run_mask))

        if np.sum(run_mask) == 0 and self.p_values is not None:
            run_mask[np.argmax(self.p_values)] = True

        (avg_betas_v[run_mask], avg_postp_v[run_mask]) = self._calculate_non_inf_betas(p, beta_tildes=self.beta_tildes[run_mask], ses=self.ses[run_mask], X_orig=self.X_orig[:,run_mask], scale_factors=self.scale_factors[run_mask], mean_shifts=self.mean_shifts[run_mask], V=None, ps=self.ps[run_mask] if self.ps is not None else None, sigma2s=self.sigma2s[run_mask] if self.sigma2s is not None else None, is_dense_gene_set=self.is_dense_gene_set[run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, betas_trace_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], debug_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], **kwargs)

        if len(avg_betas_v.shape) == 2:
            avg_betas_v = np.mean(avg_betas_v, axis=0)
            avg_postp_v = np.mean(avg_postp_v, axis=0)

        self.betas = copy.copy(avg_betas_v)
        self.betas_uncorrected = copy.copy(avg_betas_uncorrected_v)

        self.non_inf_avg_postps = copy.copy(avg_postp_v)
        self.non_inf_avg_cond_betas = copy.copy(avg_betas_v)
        self.non_inf_avg_cond_betas[avg_postp_v > 0] /= avg_postp_v[avg_postp_v > 0]

        if self.gene_sets_missing is not None:
            self.betas_missing = np.zeros(len(self.gene_sets_missing))
            self.betas_uncorrected_missing = np.zeros(len(self.gene_sets_missing))
            self.non_inf_avg_postps_missing = np.zeros(len(self.gene_sets_missing))
            self.non_inf_avg_cond_betas_missing = np.zeros(len(self.gene_sets_missing))

    def calculate_priors(self, max_gene_set_p=None, num_gene_batches=None, correct_betas_mean=True, correct_betas_var=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, p_noninf=None, run_logistic=True, max_for_linear=0.95, adjust_priors=False, tag="", **kwargs):
        if self.X_orig is None:
            bail("X is required for this operation")
        if self.betas is None:
            bail("betas are required for this operation")

        use_X = False

        assert(self.gene_sets is not None)
        max_num_gene_batches_together = 10000
        #if 0, don't use any V
        num_gene_batches_parallel = int(max_num_gene_batches_together / len(self.gene_sets))
        if num_gene_batches_parallel == 0:
            use_X = True
            log("Using low memory X instead of V in priors", TRACE)
            num_gene_batches_parallel = 1

        loco = False
        if num_gene_batches is None:
            log("Doing leave-one-chromosome-out cross validation for priors computation")
            loco = True

        if num_gene_batches is not None and num_gene_batches < 2:
            #this calculates the values for the non missing genes
            #use original X matrix here because we are rescaling betas back to those units
            priors = np.array(self.X_orig.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors)).flatten()
            self.combined_prior_Ys = None
            self.combined_prior_Ys_for_regression = None
            self.combined_prior_Ys_adj = None
            self.combined_prior_Y_ses = None
            self.combined_Ds = None
            self.batches = None
        else:

            if loco:
                if gene_loc_file is None:
                    bail("Need --gene-loc-file for --loco")

                gene_chromosomes = {}
                batches = set()
                log("Reading gene locations")
                if self.gene_to_chrom is None:
                    self.gene_to_chrom = {}
                if self.gene_to_pos is None:
                    self.gene_to_pos = {}

                with open_gz(gene_loc_file) as gene_loc_fh:
                    for line in gene_loc_fh:
                        cols = line.strip().split()
                        if len(cols) != 6:
                            bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                        gene_name = cols[5]
                        if gene_name not in self.gene_to_ind:
                            continue

                        chrom = self._clean_chrom(cols[1])
                        pos1 = int(cols[2])
                        pos2 = int(cols[3])

                        self.gene_to_chrom[gene_name] = chrom
                        self.gene_to_pos[gene_name] = (pos1,pos2)

                        batches.add(chrom)
                        gene_chromosomes[gene_name] = chrom
                batches = sorted(batches)
                num_gene_batches = len(batches)
            else:
                #need sorted genes and correlation matrix to batch genes
                if self.y_corr is None:
                    correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)
                    self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, Y_corr_m=correlation_m, skip_V=True, store_cholesky=False, skip_scale_factors=True, min_correlation=None)
                batches = range(num_gene_batches)

            gene_batch_size = int(len(self.genes) / float(num_gene_batches) + 1)
            self.batches = [None] * len(self.genes)
            priors = np.zeros(len(self.genes))

            #store a matrix of all beta_tildes across all batches
            full_matrix_shape = (len(batches), len(self.gene_sets) + (len(self.gene_sets_missing) if self.gene_sets_missing is not None else 0))
            full_beta_tildes_m = np.zeros(full_matrix_shape)
            full_ses_m = np.zeros(full_matrix_shape)
            full_z_scores_m = np.zeros(full_matrix_shape)
            full_se_inflation_factors_m = np.zeros(full_matrix_shape)
            full_p_values_m = np.zeros(full_matrix_shape)
            full_scale_factors_m = np.zeros(full_matrix_shape)
            full_ps_m = None
            if self.ps is not None:
                full_ps_m = np.zeros(full_matrix_shape)                
            full_sigma2s_m = None
            if self.sigma2s is not None:
                full_sigma2s_m = np.zeros(full_matrix_shape)                

            full_is_dense_gene_set_m = np.zeros(full_matrix_shape, dtype=bool)
            full_mean_shifts_m = np.zeros(full_matrix_shape)
            full_include_mask_m = np.zeros((len(batches), len(self.genes)), dtype=bool)
            full_priors_mask_m = np.zeros((len(batches), len(self.genes)), dtype=bool)

            #combine X_orig and X_orig missing
            revert_subset_mask = None
            if self.gene_sets_missing is not None:
                revert_subset_mask = self._unsubset_gene_sets(skip_V=True)

            for batch_ind in range(len(batches)):
                batch = batches[batch_ind]

                #specify:
                # (a) include_mask: the genes that are used for calculating beta tildes and betas for this batch
                # (b) priors_mask: the genes that we will calculate priors for
                #these are not exact complements because we may need to exlude some genes for both (i.e. a buffer)
                if loco:
                    include_mask = np.array([True] * len(self.genes))
                    priors_mask = np.array([False] * len(self.genes))
                    for i in range(len(self.genes)):
                        if self.genes[i] not in gene_chromosomes:
                            include_mask[i] = False
                            priors_mask[i] = True
                        elif gene_chromosomes[self.genes[i]] == batch:
                            include_mask[i] = False
                            priors_mask[i] = True
                        else:
                            include_mask[i] = True
                            priors_mask[i] = False
                    log("Batch %s: %d genes" % (batch, np.sum(priors_mask)))
                else:
                    begin = batch * gene_batch_size
                    end = (batch + 1) * gene_batch_size
                    if end > len(self.genes):
                        end = len(self.genes)
                    end = end - 1
                    log("Batch %d: genes %d - %d" % (batch+1, begin, end))


                    #include only genes not correlated with any in the current batch
                    include_mask = np.array([True] * len(self.genes))

                    include_mask_begin = begin - 1
                    while include_mask_begin > 0 and (begin - include_mask_begin) < len(self.y_corr) and self.y_corr[begin - include_mask_begin][include_mask_begin] > 0:
                        include_mask_begin -= 1
                    include_mask_begin += 1

                    include_mask_end = end + 1
                    while (include_mask_end - end) < len(self.y_corr) and self.y_corr[include_mask_end - end][end] > 0:
                        include_mask_end += 1
                    include_mask[include_mask_begin:include_mask_end] = False
                    include_mask_end -= 1

                    priors_mask = np.array([False] * len(self.genes))
                    priors_mask[begin:(end+1)] = True


                for i in range(len(self.genes)):
                    if priors_mask[i]:
                        self.batches[i] = batch

                #now subset Y
                Y = copy.copy(self.Y_for_regression)
                y_corr = None
                y_corr_cholesky = None
                y_corr_sparse = None

                if self.y_corr is not None:
                    y_corr = copy.copy(self.y_corr)
                    if not loco:
                        #we cannot rely on chromosome boundaries to zero out correlations, so manually do this
                        for i in range(include_mask_begin - 1, include_mask_begin - y_corr.shape[0], -1):
                            y_corr[include_mask_begin - i:,i] = 0
                    #don't need to zero out anything for include_mask_end because correlations between after end and removed are all stored inside of the removed indices
                    y_corr = y_corr[:,include_mask]

                    if self.y_corr_cholesky is not None:
                        Y = copy.copy(self.Y_fw)
                        #this is the correlation matrix we will use this batch
                        #it is a subsetted version of the self.y_corr but with the correlations with the removed genes zeroed out
                        y_corr_cholesky = self._get_y_corr_cholesky(y_corr)
                    elif self.y_corr_sparse is not None:
                        y_corr_sparse = self.y_corr_sparse[include_mask,:][:,include_mask]
                
                Y = Y[include_mask]
                y_var = np.var(Y)

                #DO WE NEED THIS??
                #y_mean = np.mean(Y)
                #Y = Y - y_mean

                (mean_shifts, scale_factors) = self._calc_X_shift_scale(self.X_orig[include_mask,:], y_corr_cholesky)

                #if some gene sets became empty!
                assert(not np.any(np.logical_and(mean_shifts != 0, scale_factors == 0)))
                mean_shifts[mean_shifts == 0] = 0
                scale_factors[scale_factors == 0] = 1

                ps = self.ps
                sigma2s = self.sigma2s
                is_dense_gene_set = self.is_dense_gene_set

                #max_gene_set_p = self.max_gene_set_p if self.max_gene_set_p is not None else 1

                Y_to_use = Y
                D = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))
                if np.max(D) > max_for_linear:
                    run_logistic = True

                #compute special beta tildes here
                if run_logistic:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig[include_mask,:], D, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)
                else:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._compute_beta_tildes(self.X_orig[include_mask,:], Y, y_var, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)

                if correct_betas_mean or correct_betas_var:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._correct_beta_tildes(beta_tildes, ses, se_inflation_factors, self.total_qc_metrics, self.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, fit=False)

                #now determine those that have too many genes removed to be accurate
                mean_reduction = float(num_gene_batches - 1) / float(num_gene_batches)
                sd_reduction = np.sqrt(mean_reduction * (1 - mean_reduction))
                reduction = mean_shifts / self.mean_shifts
                ignore_mask = reduction < mean_reduction - 3 * sd_reduction
                if sum(ignore_mask) > 0:
                    log("Ignoring %d gene sets because there are too many genes are missing from this batch" % sum(ignore_mask))
                    for ind in np.array(range(len(ignore_mask)))[ignore_mask]:
                        log("%s: %.4g remaining (vs. %.4g +/- %.4g expected)" % (self.gene_sets[ind], reduction[ind], mean_reduction, sd_reduction), TRACE)
                #also zero out anything above the p-value threshold; this is a convenience for below
                #note that p-values are still preserved though for below
                ignore_mask = np.logical_or(ignore_mask, p_values > max_gene_set_p)

                beta_tildes[ignore_mask] = 0
                ses[ignore_mask] = max(self.ses) * 100

                full_beta_tildes_m[batch_ind,:] = beta_tildes
                full_ses_m[batch_ind,:] = ses
                full_z_scores_m[batch_ind,:] = z_scores
                full_se_inflation_factors_m[batch_ind,:] = se_inflation_factors
                full_p_values_m[batch_ind,:] = p_values
                full_scale_factors_m[batch_ind,:] = scale_factors
                full_mean_shifts_m[batch_ind,:] = mean_shifts
                if full_ps_m is not None:
                    full_ps_m[batch_ind,:] = ps
                if full_sigma2s_m is not None:
                    full_sigma2s_m[batch_ind,:] = sigma2s

                full_is_dense_gene_set_m[batch_ind,:] = is_dense_gene_set
                full_include_mask_m[batch_ind,:] = include_mask
                full_priors_mask_m[batch_ind,:] = priors_mask

            #now calculate everything
            if p_noninf is None or p_noninf >= 1:
                num_gene_batches_parallel = 1
            num_calculations = int(np.ceil(num_gene_batches / num_gene_batches_parallel))
            for calc in range(num_calculations):
                begin = calc * num_gene_batches_parallel
                end = (calc + 1) * num_gene_batches_parallel
                if end > num_gene_batches:
                    end = num_gene_batches
                
                log("Running calculations for batches %d-%d" % (begin, end))

                #ensure there is at least one gene set remaining
                max_gene_set_p_v = np.min(full_p_values_m[begin:end,:], axis=1)
                #max_gene_set_p_v[max_gene_set_p_v < (self.max_gene_set_p if self.max_gene_set_p is not None else 1)] = (self.max_gene_set_p if self.max_gene_set_p is not None else 1)
                max_gene_set_p_v[max_gene_set_p_v < (max_gene_set_p if max_gene_set_p is not None else 1)] = (max_gene_set_p if max_gene_set_p is not None else 1)

                #get the include mask; any batch has p <= threshold
                new_gene_set_mask = np.max(full_p_values_m[begin:end,:].T <= max_gene_set_p_v, axis=1)
                num_gene_set_mask = np.sum(new_gene_set_mask)

                #we unsubset genes to aid in batching; this caused sigma and p to be affected
                fraction_non_missing = np.mean(new_gene_set_mask)
                missing_scale_factor = self._get_fraction_non_missing() / fraction_non_missing
                if missing_scale_factor > 1 / self.p:
                    #threshold this here. otherwise set_p will cap p but set_sigma won't cap sigma
                    missing_scale_factor = 1 / self.p
                
                #orig_sigma2 = self.sigma2
                #orig_p = self.p
                #self.set_sigma(self.sigma2 * missing_scale_factor, self.sigma_power, sigma2_osc=self.sigma2_osc)
                #self.set_p(self.p * missing_scale_factor)

                #construct the V matrix
                if not use_X:
                    V_m = np.zeros((end-begin, num_gene_set_mask, num_gene_set_mask))
                    for i,j in zip(range(begin, end),range(end-begin)):
                        include_mask = full_include_mask_m[i,:]

                        V_m[j,:,:] = self._calculate_V_internal(self.X_orig[include_mask,:][:,new_gene_set_mask], y_corr_cholesky, full_mean_shifts_m[i,new_gene_set_mask], full_scale_factors_m[i,new_gene_set_mask])
                else:
                    V_m = None

                cur_beta_tildes = full_beta_tildes_m[begin:end,:][:,new_gene_set_mask]
                cur_ses = full_ses_m[begin:end,:][:,new_gene_set_mask]
                cur_se_inflation_factors = full_se_inflation_factors_m[begin:end,:][:,new_gene_set_mask]
                cur_scale_factors = full_scale_factors_m[begin:end,:][:,new_gene_set_mask]
                cur_mean_shifts = full_mean_shifts_m[begin:end,:][:,new_gene_set_mask]
                cur_is_dense_gene_set = full_is_dense_gene_set_m[begin:end,:][:,new_gene_set_mask]
                cur_ps = None
                if full_ps_m is not None:
                    cur_ps = full_ps_m[begin:end,:][:,new_gene_set_mask]
                cur_sigma2s = None
                if full_sigma2s_m is not None:
                    cur_sigma2s = full_sigma2s_m[begin:end,:][:,new_gene_set_mask]

                #only non inf now
                (betas, avg_postp) = self._calculate_non_inf_betas(None, beta_tildes=cur_beta_tildes, ses=cur_ses, V=V_m, X_orig=self.X_orig[include_mask,:][:,new_gene_set_mask], scale_factors=cur_scale_factors, mean_shifts=cur_mean_shifts, is_dense_gene_set=cur_is_dense_gene_set, ps=cur_ps, sigma2s=cur_sigma2s, update_hyper_sigma=False, update_hyper_p=False, num_missing_gene_sets=int((1 - fraction_non_missing) * len(self.gene_sets)), **kwargs)
                if len(betas.shape) == 1:
                    betas = betas[np.newaxis,:]


                #if do inf:
                #    V = V_m[0,:,:]
                #    if self.y_corr_cholesky is None and self.y_corr_sparse is not None:
                #        V_cor = self._calculate_V_internal(self.X_orig[full_include_mask_m[begin,:],:][:,new_gene_set_mask], None, cur_mean_shifts, cur_scale_factors, y_corr_sparse=y_corr_sparse)
                #        V_inv = self._invert_sym_matrix(V)
                #    else:
                #        V_cor = None
                #        V_inv = None
                #    betas = self._calculate_inf_betas(beta_tildes=cur_beta_tildes, ses=cur_ses, V=V, V_cor=V_cor, V_inv=V_inv, se_inflation_factors=cur_se_inflation_factors, scale_factors=cur_scale_factors, is_dense_gene_set=cur_is_dense_gene_set)
                #    betas = betas[np.newaxis,:]

                for i,j in zip(range(begin, end),range(end-begin)):

                    priors[full_priors_mask_m[i,:]] = np.array(self.X_orig[full_priors_mask_m[i,:],:][:,new_gene_set_mask].dot(betas[j,:] / cur_scale_factors[j,:]))

                #now restore the p and sigma
                #self.set_sigma(orig_sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
                #self.set_p(orig_p)

            #now restore previous subsets
            self.subset_gene_sets(revert_subset_mask, keep_missing=True, skip_V=True)

        #now for the genes that were not included in X
        if self.X_orig_missing_genes is not None:
            #these can use the original betas because they were never included
            self.priors_missing = np.array(self.X_orig_missing_genes.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors))
        else:
            self.priors_missing = np.array([])

        #store in member variable
        total_mean = np.mean(np.concatenate((priors, self.priors_missing)))
        self.priors = priors - total_mean
        self.priors_missing -= total_mean

        self.calculate_priors_adj(overwrite_priors=adjust_priors)

    def calculate_priors_adj(self, overwrite_priors=False):
        if self.priors is None:
            return
        
        #do the regression
        gene_N = self.get_gene_N()
        gene_N_missing = self.get_gene_N(get_missing=True)
        all_gene_N = gene_N
        if self.genes_missing is not None:
            assert(gene_N_missing is not None)
            all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

        if self.genes_missing is not None:
            total_priors = np.concatenate((self.priors, self.priors_missing))
        else:
            total_priors = self.priors

        priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
        priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

        log("Adjusting priors with slope %.4g" % priors_slope)
        priors_adj = self.priors - priors_slope * gene_N - priors_intercept
        if overwrite_priors:
            self.priors = priors_adj
        else:
            self.priors_adj = priors_adj
        if self.genes_missing is not None:
            priors_adj_missing = self.priors_missing - priors_slope * gene_N_missing
            if overwrite_priors:
                self.priors_missing = priors_adj_missing
            else:
                self.priors_adj_missing = priors_adj_missing

    def calculate_naive_priors(self, adjust_priors=False):
        if self.X_orig is None:
            bail("X is required for this operation")
        if self.betas is None:
            bail("betas are required for this operation")
        
        self.priors = self.X_orig.dot(self.betas / self.scale_factors)

        if self.X_orig_missing_genes is not None:
            self.priors_missing = self.X_orig_missing_genes.dot(self.betas / self.scale_factors)
        else:
            self.priors_missing = np.array([])

        total_mean = np.mean(np.concatenate((self.priors, self.priors_missing)))
        self.priors -= total_mean
        self.priors_missing -= total_mean

        self.calculate_priors_adj(overwrite_priors=adjust_priors)

        if self.Y is not None:
            if self.priors is not None:
                self.combined_prior_Ys = self.priors + self.Y
            if self.priors_adj is not None:
                self.combined_prior_Ys_adj = self.priors_adj + self.Y

    def run_gibbs(self, min_num_iter=2, max_num_iter=100, num_chains=10, num_mad=3, r_threshold_burn_in=1.01, max_frac_sem=0.01, use_max_r_for_convergence=True, p_noninf=None, increase_hyper_if_betas_below=None, update_huge_scores=True, top_gene_prior=None, min_num_burn_in=10, max_num_burn_in=None, max_num_iter_betas=1100, min_num_iter_betas=10, num_chains_betas=4, r_threshold_burn_in_betas=1.01, use_max_r_for_convergence_betas=True, max_frac_sem_betas=0.01, use_mean_betas=True, sparse_frac_gibbs=0.01, sparse_max_gibbs=0.001, sparse_solution=False, sparse_frac_betas=None, pre_filter_batch_size=None, pre_filter_small_batch_size=500, max_allowed_batch_correlation=None, gauss_seidel_betas=False, gauss_seidel=False, num_gene_batches=None, num_batches_parallel=10, max_mb_X_h=200, initial_linear_filter=True, correct_betas_mean=True, correct_betas_var=True, adjust_priors=True, gene_set_stats_trace_out=None, gene_stats_trace_out=None, betas_trace_out=None, eps=0.01):

        passed_in_max_num_burn_in = max_num_burn_in
        if max_num_burn_in is None:
            max_num_burn_in = int(max_num_iter * .25)
        if max_num_burn_in >= max_num_iter:
            max_num_burn_in = int(max_num_iter * .25)


        elif num_chains < 2:
            num_chains = 2

        self._record_params({"num_chains": num_chains, "num_chains_betas": num_chains_betas, "use_mean_betas": use_mean_betas, "sparse_solution": sparse_solution, "sparse_frac": sparse_frac_gibbs, "sparse_max": sparse_max_gibbs, "sparse_frac_betas": sparse_frac_betas, "pre_filter_batch_size": pre_filter_batch_size, "max_allowed_batch_correlation": max_allowed_batch_correlation, "initial_linear_filter": initial_linear_filter, "correct_betas_mean": correct_betas_mean, "correct_betas_var": correct_betas_var, "adjust_priors": adjust_priors})

        log("Running Gibbs")

        #save all of the old values
        self.beta_tildes_orig = copy.copy(self.beta_tildes)
        self.p_values_orig = copy.copy(self.p_values)
        self.ses_orig = copy.copy(self.ses)
        self.z_scores_orig = copy.copy(self.z_scores)
        self.beta_tildes_missing_orig = copy.copy(self.beta_tildes_missing)
        self.p_values_missing_orig = copy.copy(self.p_values_missing)
        self.ses_missing_orig = copy.copy(self.ses_missing)
        self.z_scores_missing_orig = copy.copy(self.z_scores_missing)

        self.betas_orig = copy.copy(self.betas)
        self.betas_uncorrected_orig = copy.copy(self.betas_uncorrected)
        self.inf_betas_orig = copy.copy(self.inf_betas)
        self.non_inf_avg_cond_betas_orig = copy.copy(self.non_inf_avg_cond_betas)
        self.non_inf_avg_postps_orig = copy.copy(self.non_inf_avg_postps)
        self.betas_missing_orig = copy.copy(self.betas_missing)
        self.betas_uncorrected_missing_orig = copy.copy(self.betas_uncorrected_missing)
        self.inf_betas_missing_orig = copy.copy(self.inf_betas_missing)
        self.non_inf_avg_cond_betas_missing_orig = copy.copy(self.non_inf_avg_cond_betas_missing)
        self.non_inf_avg_postps_missing_orig = copy.copy(self.non_inf_avg_postps_missing)

        self.Y_orig = copy.copy(self.Y)
        self.Y_for_regression_orig = copy.copy(self.Y_for_regression)
        self.Y_w_orig = copy.copy(self.Y_w)
        self.Y_fw_orig = copy.copy(self.Y_fw)
        self.priors_orig = copy.copy(self.priors)
        self.priors_adj_orig = copy.copy(self.priors_adj)
        self.priors_missing_orig = copy.copy(self.priors_missing)

        self.priors_adj_missing_orig = copy.copy(self.priors_adj_missing)


        #we always update correlation relative to the original one
        y_var_orig = np.var(self.Y_for_regression)

        #set up constants throughout the loop

        Y_to_use = self.Y_for_regression_orig
        bf_orig = np.exp(Y_to_use)

        bf_orig_raw = np.exp(self.Y_orig)

        #conditional variance of Y given beta: calculate residuals given priors
        priors_guess = np.array(self.X_orig.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors))

        Y_resid = np.var(self.Y_for_regression_orig - priors_guess)
        Y_cond_var = Y_resid

        if top_gene_prior is not None:
            if top_gene_prior <= 0 or top_gene_prior >= 1:
                bail("--top-gene-prior needs to be in (0,1)")
            Y_total_var = self.convert_prior_to_var(top_gene_prior, len(self.genes))
            Y_cond_var = Y_total_var - self.get_sigma2(convert_sigma_to_external_units=True) * np.mean(self.get_gene_N())
            if Y_cond_var < 0:
                #minimum value
                Y_cond_var = 0.1
            log("Setting Y cond var=%.4g (total var = %.4g) given top gene prior of %.4g" % (Y_cond_var, Y_total_var, top_gene_prior))

        Y_cond_sd = np.sqrt(Y_cond_var)


        #this is the density of the relative (log) prior odds

        bf_orig_m = np.tile(bf_orig, num_chains).reshape(num_chains, len(bf_orig))
        log_bf_m = np.log(bf_orig_m)
        log_bf_uncorrected_m = np.log(bf_orig_m)

        bf_orig_raw_m = np.tile(bf_orig_raw, num_chains).reshape(num_chains, len(bf_orig_raw))
        log_bf_raw_m = np.log(bf_orig_raw_m)

        compute_Y_raw = np.any(~np.isclose(log_bf_m, log_bf_raw_m))

        #we will adjust this to preserve the original probabilities if requested
        cur_background_log_bf_v = np.tile(self.background_log_bf, num_chains)

        def __density_fun(x, loc, scale, bf=bf_orig, background_log_bf=self.background_log_bf, do_expected=False):
            if type(x) == np.ndarray:
                prob = np.ones(x.shape)
                okay_mask = x < 10
                #we need absolute odds (not relative) for this calculation so add in background_log_bf
                x_odds = np.exp(x[okay_mask] + background_log_bf)
                prob[okay_mask] =  x_odds / (1 + x_odds)
            else:
                if x < 10:
                    x_odds = np.exp(x + background_log_bf)
                    prob = x_odds / (1 + x_odds)
                else:
                    prob = 1
        
            density = (bf * prob + (1 - prob)) * scipy.stats.norm.pdf(x, loc=loc, scale=scale)
            if do_expected:
                return np.dstack((density.T, x * density.T, np.square(x) * density.T)).T
            else:
                return density

        def __outlier_resistant_mean(sum_m, num_sum_m, outlier_mask_m=None):
            if outlier_mask_m is None:

                self._record_param("mad_threshold", num_mad)

                #1. calculate mean values for each chain (divide by number -- make sure it is correct; may not be num_avg_Y)
                chain_means_m = sum_m / num_sum_m

                #2. calculate median values across chains (one number per gene set/gene)
                medians_v = np.median(chain_means_m, axis=0)

                #3. calculate abs(difference) between each chain and median (one value per chain/geneset)
                mad_m = np.abs(chain_means_m - medians_v)

                #4. calculate median of abs(difference) across chains (one number per gene set/gene)
                mad_median_v = np.median(mad_m, axis=0)

                #5. mask any chain that is more than 3 median(abs(difference)) from median
                outlier_mask_m = chain_means_m > medians_v + num_mad * mad_median_v

            #6. take average only across chains that are not outliers
            num_sum_v = np.sum(~outlier_mask_m, axis=0)

            #should never happen but just in case
            num_sum_v[num_sum_v == 0] = 1

            #7. to do this, zero out outlier chains, then sum them, then divide by number of outliers
            copy_sum_m = copy.copy(sum_m)
            copy_sum_m[outlier_mask_m] = 0
            avg_v = np.sum(copy_sum_m / num_sum_m, axis=0) / num_sum_v
            
            return (outlier_mask_m, avg_v)


        #initialize Y

        if self.y_corr_cholesky is not None:
            bail("GLS not implemented yet for Gibbs sampling!")

        #dimensions of matrices are (num_chains, num_gene_sets)

        num_full_gene_sets = len(self.gene_sets)
        if self.gene_sets_missing is not None:
            num_full_gene_sets += len(self.gene_sets_missing)

        beta_tilde_outlier_z_threshold = None

        #this loop checks if the gibbs loop was successful

        max_num_restarts = 20
        num_p_increases = 0

        for num_restarts in range(0,max_num_restarts+1):

            #by default it succeeded
            gibbs_good = True

            #for increasing p option
            p_scale_factor = 1 - np.log(self.p)/(2 * np.log(10))
            num_before_checking_p_increase = max(min_num_iter, min_num_burn_in)
            if increase_hyper_if_betas_below is not None and num_before_checking_p_increase > min_num_iter:
                #make sure that we always trigger this check before breaking
                min_num_iter = num_before_checking_p_increase

            self._record_param("num_gibbs_restarts", num_restarts, overwrite=True)
            if num_restarts > 0:
                log("Gibbs restart %d" % num_restarts)

            num_restarts += 1

            burn_in_phase_beta_v = np.full(num_full_gene_sets, True)

            #set_to_zero_v = np.zeros(num_full_gene_sets)
            #avg_full_betas_sample_v = np.zeros(num_full_gene_sets)
            #avg_full_postp_sample_v = np.zeros(num_full_gene_sets)


            #sum of values for each chain
            #TODO: add in values for everything that has sum

            full_betas_m_shape = (num_chains, num_full_gene_sets)
            sum_betas_m = np.zeros(full_betas_m_shape)
            sum_betas2_m = np.zeros(full_betas_m_shape)
            sum_betas_uncorrected_m = np.zeros(full_betas_m_shape)
            sum_postp_m = np.zeros(full_betas_m_shape)
            sum_beta_tildes_m = np.zeros(full_betas_m_shape)
            sum_z_scores_m = np.zeros(full_betas_m_shape)
            num_sum_beta_m = np.zeros(full_betas_m_shape)

            Y_m_shape = (num_chains, len(self.Y_for_regression))
            burn_in_phase_Y_v = np.full(Y_m_shape[1], True)
            sum_Ys_m = np.zeros(Y_m_shape)
            sum_Ys2_m = np.zeros(Y_m_shape)
            sum_Y_raws_m = np.zeros(Y_m_shape)
            sum_log_pos_m = np.zeros(Y_m_shape)
            sum_log_pos2_m = np.zeros(Y_m_shape)
            sum_log_po_raws_m = np.zeros(Y_m_shape)
            sum_priors_m = np.zeros(Y_m_shape)
            sum_Ds_m = np.zeros(Y_m_shape)
            sum_D_raws_m = np.zeros(Y_m_shape)
            sum_bf_orig_m = np.zeros(Y_m_shape)
            sum_bf_uncorrected_m = np.zeros(Y_m_shape)
            sum_bf_orig_raw_m = np.zeros(Y_m_shape)
            num_sum_Y_m = np.zeros(Y_m_shape)

            #sums across all iterations, not just converged
            all_sum_betas_m = np.zeros(full_betas_m_shape)
            all_sum_betas2_m = np.zeros(full_betas_m_shape)
            all_sum_z_scores_m = np.zeros(full_betas_m_shape)
            all_sum_z_scores2_m = np.zeros(full_betas_m_shape)
            all_num_sum_m = np.zeros(full_betas_m_shape)

            all_sum_Ys_m = np.zeros(Y_m_shape)
            all_sum_Ys2_m = np.zeros(Y_m_shape)

            #num_sum = 0

            #sum_Ys_post_m = np.zeros(Y_m_shape)
            #sum_Ys2_post_m = np.zeros(Y_m_shape)
            #num_sum_post = 0

            #sum across all chains

            #avg_betas = np.zeros(num_full_gene_sets)
            #avg_betas2 = np.zeros(num_full_gene_sets)
            #avg_betas_uncorrected = np.zeros(num_full_gene_sets)

            #avg_postp = np.zeros(num_full_gene_sets)
            #avg_beta_tildes = np.zeros(num_full_gene_sets)
            #avg_z_scores = np.zeros(num_full_gene_sets)
            #avg_Ys = np.zeros(len(self.Y))
            #avg_Ys2 = np.zeros(len(self.Y))
            #avg_log_pos = np.zeros(len(self.Y))
            #avg_log_pos2 = np.zeros(len(self.Y))
            #avg_priors = np.zeros(len(self.Y))
            #avg_Ds = np.zeros(len(self.Y))
            #avg_bf_orig = np.zeros(len(self.Y))

            #num_avg_beta = np.zeros(num_full_gene_sets)
            #num_avg_Y = np.zeros(len(self.Y))

            #initialize the priors
            priors_sample_m = np.zeros(Y_m_shape)
            priors_mean_m = np.zeros(Y_m_shape)

            priors_percentage_max_sample_m = np.zeros(Y_m_shape)
            priors_percentage_max_mean_m = np.zeros(Y_m_shape)
            priors_adjustment_sample_m = np.zeros(Y_m_shape)
            priors_adjustment_mean_m = np.zeros(Y_m_shape)

            priors_for_Y_m = priors_sample_m
            priors_percentage_max_for_Y_m = priors_percentage_max_sample_m
            priors_adjustment_for_Y_m = priors_adjustment_sample_m
            if use_mean_betas:
                priors_for_Y_m = priors_mean_m                
                priors_percentage_max_for_Y_m = priors_percentage_max_mean_m
                priors_adjustment_for_Y_m = priors_adjustment_mean_m

            num_genes_missing = 0
            if self.genes_missing is not None:
                num_genes_missing = len(self.genes_missing)

            sum_priors_missing_m = np.zeros((num_chains, num_genes_missing))

            sum_Ds_missing_m = np.zeros((num_chains, num_genes_missing))

            #avg_priors_missing = np.zeros(num_genes_missing)
            #avg_Ds_missing = np.zeros(num_genes_missing)

            priors_missing_sample_m = np.zeros(sum_priors_missing_m.shape)
            priors_missing_mean_m = np.zeros(sum_priors_missing_m.shape)
            num_sum_priors_missing_m = np.zeros(sum_priors_missing_m.shape)

            if gene_set_stats_trace_out is not None:

                gene_set_stats_trace_fh = open_gz(gene_set_stats_trace_out, 'w')
                gene_set_stats_trace_fh.write("It\tChain\tGene_Set\tbeta_tilde\tP\tZ\tSE\tbeta_uncorrected\tbeta\tpostp\tbeta_tilde_outlier_z\tR\tSEM\n")
            if gene_stats_trace_out is not None:
                gene_stats_trace_fh = open_gz(gene_stats_trace_out, 'w')
                gene_stats_trace_fh.write("It\tChain\tGene\tprior\tcombined\tlog_bf\tD\tpercent_top\tadjust\n")

            #TEMP STUFF
            only_genes = None
            only_gene_sets = None

            if self.gene_sets_missing is not None:
                revert_subset_mask = self._unsubset_gene_sets(skip_V=True)

            prev_Ys_m = None
            #cache this
            X_hstacked = None
            stack_batch_size = num_chains + 1
            if num_chains > 1:

                X_size_mb = self._get_X_size_mb()
                X_h_size_mb =  num_chains * X_size_mb
                if X_h_size_mb <= max_mb_X_h:
                    X_hstacked = sparse.hstack([self.X_orig] * num_chains)
                else:
                    stack_batch_size = int(max_mb_X_h / X_size_mb)
                    if stack_batch_size == 0:
                        stack_batch_size = 1
                    log("Not building X_hstacked, size would be %d > %d; will instead run %d chains at a time" % (X_h_size_mb, max_mb_X_h, stack_batch_size))
                    X_hstacked = sparse.hstack([self.X_orig] * stack_batch_size)
            else:
                X_hstacked = self.X_orig

            num_stack_batches = int(np.ceil(num_chains / stack_batch_size))

            X_all = self.X_orig
            if self.genes_missing is not None:
                X_all = sparse.vstack((self.X_orig, self.X_orig_missing_genes))

            for iteration_num in range(max_num_iter):

                log("Beginning Gibbs iteration %d" % (iteration_num+1))
                self._record_param("num_gibbs_iter", iteration_num, overwrite=True)

                log("Sampling new Ys")

                log("Setting logistic Ys", TRACE)

                Y_sample_m = priors_for_Y_m + log_bf_m
                Y_raw_sample_m = priors_for_Y_m + log_bf_raw_m
                y_var = np.var(Y_sample_m, axis=1)

                #if adjust_background_prior:
                #    #get the original mean bf
                #    background_log_prior_scale_factor = np.mean(Y_sample_m, axis=1) - np.mean(log_bf_orig_m, axis=1)
                #    cur_background_log_bf_v = self.background_log_bf + background_log_prior_scale_factor
                #    cur_background_bf_v = np.exp(cur_background_log_bf_v)
                #    log("Adjusting background priors to %.4g-%.4g" % (np.min(cur_background_bf_v / (1 + cur_background_bf_v)), np.max(cur_background_bf_v / (1 + cur_background_bf_v))))

                #threshold in case things go off the rails
                max_log = 15

                cur_log_bf_m = Y_sample_m.T + cur_background_log_bf_v
                cur_log_bf_m[cur_log_bf_m > max_log] = max_log
                bf_sample_m = np.exp(cur_log_bf_m).T

                cur_log_bf_raw_m = Y_raw_sample_m.T + cur_background_log_bf_v
                cur_log_bf_raw_m[cur_log_bf_raw_m > max_log] = max_log
                bf_raw_sample_m = np.exp(cur_log_bf_raw_m).T

                #FIXME: do we really need to add in background_log_bf and then subtract it below? Surely there must be a way to do these calculations independent of background_log_bf? Can check by seeing if results are invariant to background_log_bf
                max_D = 1-1e-5
                min_D = 1e-5

                D_sample_m = bf_sample_m / (1 + bf_sample_m)
                D_sample_m[D_sample_m > max_D] = max_D
                D_sample_m[D_sample_m < min_D] = min_D
                log_po_sample_m = np.log(D_sample_m/(1-D_sample_m))

                D_raw_sample_m = bf_raw_sample_m / (1 + bf_raw_sample_m)
                D_raw_sample_m[D_raw_sample_m > max_D] = max_D
                D_raw_sample_m[D_raw_sample_m < min_D] = min_D
                log_po_raw_sample_m = np.log(D_raw_sample_m/(1-D_raw_sample_m))

                #if center_combined:
                #    #recenter the log_pos and Ds as well
                #    #the "combined missing" is just the priors
                #    log_po_sample_total_m = np.hstack((log_po_sample_m, priors_missing_sample_m))
                #    total_po_mean_v = np.mean(log_po_sample_total_m, axis=1)
                #    log_po_sample_m = (log_po_sample_m.T - total_po_mean_v).T
                #    bf_sample_m = np.exp(log_po_sample_m.T + cur_background_log_bf_v).T
                #    D_sample_m = bf_sample_m / (1 + bf_sample_m)
                #else:
                #    log("Not centering combined")


                #We must normalize Y_sample_m for the compute beta tildes!
                #FIXME: this led to a bug and should be updated to prevent errors in the future
                #TESTING removal of this standardization
                #Y_sample_m = (Y_sample_m.T - np.mean(Y_sample_m, axis=1)).T

                #var(Y) = E[var(Y|S,beta)] + var(E[Y|S,beta])
                #First term can be estimated from the gibbs samples
                #Second term is just Y_cond_var (to a first approximation), or more accurately the term in the integral from gauss
                #Third term is the term we estimate from the Gauss seidel regression
                #So: if we use
                #y_var = np.var(Y_sample_m, axis=1)
                #the term we want, var(Y|S,beta) is being overestimated for gibbs, underestimated for gauss seidel
                #Let's first try Gauss seidel with correction, then try Gibbs with the first approximation (see the difference)
                #This is what is implemented above in Y_var_m -- gives conditional variance of each Y
                #Taking mean of this is our other estimate

                #sample from beta

                #combine X_orig and X_orig missing?

                #TODO: y_corr_sparse needs to be reduced due to y_var (it is larger here than it is above)
                #TODO: if decide calculations depend on chain, then also need to update compute_beta_tildes to return matrix of se_inflation_factors
                y_corr_sparse = None
                if self.y_corr_sparse is not None:

                    log("Adjusting correlation matrix")

                    y_corr_sparse = copy.copy(self.y_corr_sparse)

                    #lower the correlation to account for the 
                    y_corr_sparse = y_corr_sparse.multiply(y_var_orig)

                    #new variances
                    new_y_sd = np.sqrt(np.square(np.mean(priors_for_Y_m, axis=0)) + y_var_orig)[np.newaxis,:]

                    y_corr_sparse = y_corr_sparse.multiply(1/new_y_sd.T)
                    y_corr_sparse = y_corr_sparse.multiply(1/new_y_sd)
                    y_corr_sparse.setdiag(1)

                    y_corr_sparse = y_corr_sparse.tocsc()


                #NOW ONTO GENE SETS

                def __get_gene_set_mask(uncorrected_betas_mean_m, uncorrected_betas_sample_m, p_values_m, sparse_frac=0.01, sparse_max=0.001):
                    #if desired, add back in option to set to sample
                    #uncorrected_betas_m = uncorrected_betas_sample_m

                    uncorrected_betas_m = uncorrected_betas_mean_m

                    gene_set_mask_m = uncorrected_betas_m != 0

                    if sparse_frac is not None:
                        #this triggers three things
                        #1. Only gene sets above this threshold are considered for full analysis
                        gene_set_mask_m = np.logical_and(gene_set_mask_m, (np.abs(uncorrected_betas_m).T >= sparse_frac * np.max(np.abs(uncorrected_betas_m), axis=1)).T)
                        #2. Or if below this max cap
                        gene_set_mask_m = np.logical_and(gene_set_mask_m, (np.abs(uncorrected_betas_m).T >= sparse_max).T)

                        #3. The uncorrected values for sampling next iteration are also zeroed out
                        uncorrected_betas_sample_m[~gene_set_mask_m] = 0
                        #4. The mean values (which are added to the estimate) are also zeroed out
                        uncorrected_betas_mean_m[~gene_set_mask_m] = 0

                    if np.sum(gene_set_mask_m) == 0:
                        gene_set_mask_m = p_values_m <= np.min(p_values_m)
                    return gene_set_mask_m


                full_scale_factors_m = np.tile(self.scale_factors, num_chains).reshape((num_chains, len(self.scale_factors)))
                full_mean_shifts_m = np.tile(self.mean_shifts, num_chains).reshape((num_chains, len(self.mean_shifts)))
                full_is_dense_gene_set_m = np.tile(self.is_dense_gene_set, num_chains).reshape((num_chains, len(self.is_dense_gene_set)))
                full_ps_m = None
                if self.ps is not None:
                    full_ps_m = np.tile(self.ps, num_chains).reshape((num_chains, len(self.ps)))
                full_sigma2s_m = None
                if self.sigma2s is not None:
                    full_sigma2s_m = np.tile(self.sigma2s, num_chains).reshape((num_chains, len(self.sigma2s)))

                #we have to keep local replicas here because unsubset does not restore the original order, which would break full_beta_tildes and full_betas

                p_sample_m = copy.copy(Y_sample_m)

                pre_gene_set_filter_mask = None
                full_z_cur_beta_tildes_m = np.zeros(full_betas_m_shape)

                #have to do logistic or else doesn't converge
                #if run_logistic:
                if True:
                    if not gauss_seidel:
                        log("Sampling Ds for logistic", TRACE)
                        p_sample_m = np.zeros(D_sample_m.shape)
                        p_sample_m[np.random.random(D_sample_m.shape) < D_sample_m] = 1

                    else:
                        log("Setting Ds to mean probabilities", TRACE)
                        p_sample_m = D_sample_m

                    if initial_linear_filter:
                        (linear_beta_tildes_m, linear_ses_m, linear_z_scores_m, linear_p_values_m, linear_se_inflation_factors_m) = self._compute_beta_tildes(self.X_orig, Y_sample_m, y_var, self.scale_factors, self.mean_shifts, resid_correlation_matrix=y_corr_sparse)
                        (linear_uncorrected_betas_sample_m, linear_uncorrected_postp_sample_m, linear_uncorrected_betas_mean_m, linear_uncorrected_postp_mean_m) = self._calculate_non_inf_betas(assume_independent=True, initial_p=None, beta_tildes=linear_beta_tildes_m, ses=linear_ses_m, V=None, X_orig=None, scale_factors=full_scale_factors_m, mean_shifts=full_mean_shifts_m, is_dense_gene_set=full_is_dense_gene_set_m, ps=full_ps_m, sigma2s=full_sigma2s_m, return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, debug_gene_sets=self.gene_sets)
                        pre_gene_set_filter_mask_m = __get_gene_set_mask(linear_uncorrected_betas_mean_m, linear_uncorrected_betas_sample_m, linear_p_values_m, sparse_frac=sparse_frac_gibbs, sparse_max=sparse_max_gibbs)
                        pre_gene_set_filter_mask = np.any(pre_gene_set_filter_mask_m, axis=0)

                        log("Filtered down to %d gene sets using linear pre-filtering" % np.sum(pre_gene_set_filter_mask))
                    else:
                        pre_gene_set_filter_mask = np.full(full_beta_tildes_m.shape[1], True)


                    full_beta_tildes_m = np.zeros(full_betas_m_shape)
                    full_ses_m = np.zeros(full_betas_m_shape)
                    full_z_scores_m = np.zeros(full_betas_m_shape)
                    full_p_values_m = np.zeros(full_betas_m_shape)
                    se_inflation_factors_m = np.zeros(full_betas_m_shape)
                    full_alpha_tildes_m = np.zeros(full_betas_m_shape)
                    diverged_m = np.full(full_betas_m_shape, False)

                    for batch in range(num_stack_batches):
                        begin = batch * stack_batch_size
                        end = (batch + 1) * stack_batch_size
                        if end > num_chains:
                            end = num_chains

                        log("Batch %d: chains %d-%d" % (batch, begin, end), TRACE)
                        num_cur_stack = (end - begin)
                        if num_cur_stack == stack_batch_size:
                            cur_X_hstacked = X_hstacked
                        else:
                            cur_X_hstacked = sparse.hstack([self.X_orig] * num_cur_stack)

                        stack_mask = np.tile(pre_gene_set_filter_mask, num_cur_stack)

                        (full_beta_tildes_m[begin:end,pre_gene_set_filter_mask], full_ses_m[begin:end,pre_gene_set_filter_mask], full_z_scores_m[begin:end,pre_gene_set_filter_mask], full_p_values_m[begin:end,pre_gene_set_filter_mask], init_se_inflation_factors_m, full_alpha_tildes_m[begin:end,pre_gene_set_filter_mask], diverged_m[begin:end,pre_gene_set_filter_mask]) = self._compute_logistic_beta_tildes(self.X_orig[:,pre_gene_set_filter_mask], p_sample_m[begin:end,:], self.scale_factors[pre_gene_set_filter_mask], self.mean_shifts[pre_gene_set_filter_mask], resid_correlation_matrix=y_corr_sparse, X_stacked=cur_X_hstacked[:,stack_mask])

                        full_ses_m[begin:end,~pre_gene_set_filter_mask] = 100
                        full_p_values_m[begin:end,~pre_gene_set_filter_mask] = 1

                        if init_se_inflation_factors_m is not None:
                            se_inflation_factors_m[begin:end,pre_gene_set_filter_mask] = init_se_inflation_factors_m
                        else:
                            se_inflation_factors_m = None

                    #old unconditional one; shouldn't be necessary
                    #else:
                    #    (full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m, full_alpha_tildes_m, diverged_m) = self._compute_logistic_beta_tildes(self.X_orig, p_sample_m, self.scale_factors, self.mean_shifts, resid_correlation_matrix=y_corr_sparse, X_stacked=X_hstacked)
                    #    pre_gene_set_filter_mask = np.full(full_beta_tildes_m.shape[1], True)

                    #calculate whether the sample was an outlier

                    if beta_tilde_outlier_z_threshold is not None:
                        self._record_param("beta_tilde_outlier_z_threshold", beta_tilde_outlier_z_threshold)

                        mean_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        mean2_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        se_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        z_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        num_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)

                        #calculate mean_m as mean ignoring the current chain
                        mean_for_outlier_m = np.sum(all_sum_z_scores_m, axis=0) - all_sum_z_scores_m
                        mean2_for_outlier_m = np.sum(all_sum_z_scores2_m, axis=0) - all_sum_z_scores2_m
                        num_for_outlier_m = np.sum(all_num_sum_m, axis=0) - all_num_sum_m
                        num_for_outlier_non_zero_mask_m = num_for_outlier_m > 0
                        mean_for_outlier_m[num_for_outlier_non_zero_mask_m] = mean_for_outlier_m[num_for_outlier_non_zero_mask_m] / num_for_outlier_m[num_for_outlier_non_zero_mask_m]
                        mean2_for_outlier_m[num_for_outlier_non_zero_mask_m] = mean2_for_outlier_m[num_for_outlier_non_zero_mask_m] / num_for_outlier_m[num_for_outlier_non_zero_mask_m]

                        se_for_outlier_m[num_for_outlier_non_zero_mask_m] = np.sqrt(mean2_for_outlier_m[num_for_outlier_non_zero_mask_m] - np.square(mean_for_outlier_m[num_for_outlier_non_zero_mask_m]))

                        se_for_outlier_mask_m = se_for_outlier_m > 0
                        full_z_cur_beta_tildes_m[se_for_outlier_mask_m] = (full_z_scores_m[se_for_outlier_mask_m] - mean_for_outlier_m[se_for_outlier_mask_m]) / se_for_outlier_m[se_for_outlier_mask_m]
                        outlier_mask_m = full_z_cur_beta_tildes_m > beta_tilde_outlier_z_threshold

                        if np.sum(outlier_mask_m) > 0:
                            log("Detected %d outlier gene sets: %s" % (np.sum(outlier_mask_m), ",".join([self.gene_sets[i] for i in np.where(np.any(outlier_mask_m, axis=0))[0]])),DEBUG)

                            outlier_control = False
                            if outlier_control:
                                #inflate them
                                #full_beta_tildes_m[outlier_mask_m] / full_ses_m[outlier_mask_m] - mean_for_outlier_m[outlier_mask_m] = beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m]
                                #full_beta_tildes_m[outlier_mask_m] / full_ses_m[outlier_mask_m] = mean_for_outlier_m[outlier_mask_m] + beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m]
                                #full_beta_tildes_m[outlier_mask_m] = full_ses_m[outlier_mask_m] * (mean_for_outlier_m[outlier_mask_m] + beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m])

                                new_ses_m = np.abs(full_beta_tildes_m[outlier_mask_m] / (mean_for_outlier_m[outlier_mask_m] + beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m]))
                                log("Inflated ses from %.3g - %.3g" % (np.min(new_ses_m / full_ses_m[outlier_mask_m]), np.max(new_ses_m / full_ses_m[outlier_mask_m])), TRACE)

                                full_ses_m[outlier_mask_m] = new_ses_m
                                full_z_scores_m[outlier_mask_m] = full_beta_tildes_m[outlier_mask_m] / new_ses_m
                                full_p_values_m[outlier_mask_m] = 2*scipy.stats.norm.cdf(-np.abs(full_z_scores_m[outlier_mask_m]))

                            else:

                                #first check if we need to reset entire chains
                                num_outliers = np.sum(outlier_mask_m, axis=1)
                                frac_outliers = num_outliers / outlier_mask_m.shape[1]
                                chain_outlier_frac_threshold = 0.0005
                                outlier_chains = frac_outliers > chain_outlier_frac_threshold
                                for outlier_chain in np.where(outlier_chains)[0]:
                                    log("Detected entire chain %d as an outlier since it had %d (%.4g fraction) outliers" % (outlier_chain+1,num_outliers[outlier_chain], frac_outliers[outlier_chain]), DEBUG)
                                    if np.sum(~outlier_chains) > 0:
                                        replacement_chain = np.random.choice(np.where(~outlier_chains)[0], size=1)
                                        for matrix in [full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m, full_alpha_tildes_m, diverged_m]:
                                            if matrix is not None:
                                                matrix[outlier_chain,:] = matrix[replacement_chain,:]
                                        outlier_mask_m[outlier_chain,:] = False

                                    else:
                                        log("Everything was an outlier chain so doing nothing", DEBUG)

                                outlier_gene_sets = np.any(outlier_mask_m, axis=0)
                                for outlier_gene_set in np.where(outlier_gene_sets)[0]:
                                    non_outliers = ~outlier_mask_m[:,outlier_gene_set]
                                    if np.sum(non_outliers) > 0:
                                        log("Resetting %s chain %s; beta_tilde=%s and z=%s" % (self.gene_sets[outlier_gene_set], np.where(outlier_mask_m[:,outlier_gene_set])[0] + 1, full_beta_tildes_m[outlier_mask_m[:,outlier_gene_set],outlier_gene_set], full_z_cur_beta_tildes_m[outlier_mask_m[:,outlier_gene_set],outlier_gene_set]),DEBUG)

                                        replacement_chains = np.random.choice(np.where(non_outliers)[0], size=np.sum(outlier_mask_m[:,outlier_gene_set]))
                                        for matrix in [full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m, full_alpha_tildes_m, diverged_m]:
                                            if matrix is not None:
                                                matrix[outlier_mask_m[:,outlier_gene_set],outlier_gene_set] = matrix[replacement_chains,outlier_gene_set]

                                        #now make the Z threshold more lenient
                                        #beta_tilde_outlier_z_threshold[outlier_gene_set] = -scipy.stats.norm.ppf(scipy.stats.norm.cdf(-np.abs(beta_tilde_outlier_z_threshold[outlier_gene_set])) / 10)
                                        #log("New threshold is z=%.4g" % beta_tilde_outlier_z_threshold[outlier_gene_set], TRACE)

                                    else:
                                        log("Everything was an outlier for gene set %s so doing nothing" % (self.gene_sets[outlier_gene_set]), DEBUG)

                    else:
                        full_z_cur_beta_tildes_m = np.zeros(full_beta_tildes_m.shape)


                    if np.sum(diverged_m) > 0:
                        for c in range(diverged_m.shape[0]):
                            if np.sum(diverged_m[c,:] > 0):
                                for p in np.nditer(np.where(diverged_m[c,:])):
                                    log("Chain %d: gene set %s diverged" % (c+1, self.gene_sets[p]), DEBUG)
                #else:
                #    (full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m) = self._compute_beta_tildes(self.X_orig, Y_sample_m, y_var, self.scale_factors, self.mean_shifts, resid_correlation_matrix=y_corr_sparse)

                if correct_betas_mean or correct_betas_var:
                    (full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m) = self._correct_beta_tildes(full_beta_tildes_m, full_ses_m, se_inflation_factors_m, self.total_qc_metrics, self.total_qc_metrics_directions, correct_mean=correct_betas_mean, correct_var=correct_betas_var, fit=False)


                #now write the gene stats trace

                if gene_stats_trace_out is not None:
                    log("Writing gene stats trace", TRACE)
                    for chain_num in range(num_chains):
                        for i in range(len(self.genes)):
                            if only_genes is None or self.genes[i] in only_genes:
                                gene_stats_trace_fh.write("%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num+1, chain_num+1, self.genes[i], priors_for_Y_m[chain_num,i], Y_sample_m[chain_num,i], log_bf_m[chain_num,i], p_sample_m[chain_num,i], priors_percentage_max_for_Y_m[chain_num,i], priors_adjustment_for_Y_m[chain_num,i]))
                        #if self.genes_missing is not None:
                        #    for i in range(len(self.genes_missing)):
                        #        if only_genes is None or self.genes[i] in only_genes:
                        #            gene_stats_trace_fh.write("%d\t%d\t%s\t%.4g\t%s\t%s\t%s\n" % (iteration_num+1, chain_num+1, self.genes_missing[i], priors_missing_sample_m[chain_num,i], "NA", "NA", "NA"))

                    gene_stats_trace_fh.flush()

                (uncorrected_betas_sample_m, uncorrected_postp_sample_m, uncorrected_betas_mean_m, uncorrected_postp_mean_m) = self._calculate_non_inf_betas(assume_independent=True, initial_p=None, beta_tildes=full_beta_tildes_m, ses=full_ses_m, V=None, X_orig=None, scale_factors=full_scale_factors_m, mean_shifts=full_mean_shifts_m, is_dense_gene_set=full_is_dense_gene_set_m, ps=full_ps_m, sigma2s=full_sigma2s_m, return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, debug_gene_sets=self.gene_sets)

                #initial values to use
                #we will overwrite these with the corrected betas
                #but, if we decide to filter them out due to sparsity, we'll persist with the (small) uncorrected values
                default_betas_sample_m = copy.copy(uncorrected_betas_sample_m)
                default_postp_sample_m = copy.copy(uncorrected_postp_sample_m)
                default_betas_mean_m = copy.copy(uncorrected_betas_mean_m)
                default_postp_mean_m = copy.copy(uncorrected_postp_mean_m)

                #filter back down based on max p?
                gene_set_mask_m = np.full(full_p_values_m.shape, True)

                #no longer an option
                #if self.max_gene_set_p is not None and use_orig_gene_set_p:
                #    gene_set_mask_m = np.tile(self.p_values_orig <= self.max_gene_set_p, num_chains).reshape((num_chains, len(gene_set_mask)))

                gene_set_mask_m = __get_gene_set_mask(uncorrected_betas_mean_m, uncorrected_betas_sample_m, full_p_values_m, sparse_frac=sparse_frac_gibbs, sparse_max=sparse_max_gibbs)
                any_gene_set_mask = np.any(gene_set_mask_m, axis=0)
                if pre_filter_batch_size is not None and np.sum(any_gene_set_mask) > pre_filter_batch_size:
                    num_batches = self._get_num_X_blocks(self.X_orig[:,any_gene_set_mask], batch_size=pre_filter_small_batch_size)
                    if num_batches > 1:
                        #try to run with small batches to see if we can zero out more
                        gene_set_block_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,any_gene_set_mask], mean_shifts=self.mean_shifts[any_gene_set_mask], scale_factors=self.scale_factors[any_gene_set_mask], find_correlated_instead=pre_filter_small_batch_size)

                        if len(gene_set_block_masks) > 0:
                            if np.sum(gene_set_block_masks[-1]) == 1 and len(gene_set_block_masks) > 1:
                                #merge singletons at the end into the one before
                                gene_set_block_masks[-2][gene_set_block_masks[-1]] = True
                                gene_set_block_masks = gene_set_block_masks[:-1]
                            if len(gene_set_block_masks) > 1 and np.sum(gene_set_block_masks[0]) > 1:
                                #find map of indices to original indices
                                V_data = []
                                V_rows = []
                                V_cols = []
                                for gene_set_block_mask in gene_set_block_masks:
                                    V_block = self._calculate_V_internal(self.X_orig[:,any_gene_set_mask][:,gene_set_block_mask], self.y_corr_cholesky, self.mean_shifts[any_gene_set_mask][gene_set_block_mask], self.scale_factors[any_gene_set_mask][gene_set_block_mask])
                                    orig_indices = np.where(gene_set_block_mask)[0]
                                    V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
                                    V_cols += list(np.tile(orig_indices, V_block.shape[0]))
                                    V_data += list(V_block.ravel())

                                V_sparse = sparse.csc_matrix((V_data, (V_rows, V_cols)), shape=(np.sum(any_gene_set_mask), np.sum(any_gene_set_mask)))
                                log("Running %d blocks to check for zeros..." % len(gene_set_block_masks),DEBUG)
                                (half_corrected_betas_sample_m, half_corrected_postp_sample_m, half_corrected_betas_mean_m, half_corrected_postp_mean_m) = self._calculate_non_inf_betas(initial_p=None, beta_tildes=full_beta_tildes_m[:,any_gene_set_mask], ses=full_ses_m[:,any_gene_set_mask], V=V_sparse, X_orig=None, scale_factors=full_scale_factors_m[:,any_gene_set_mask], mean_shifts=full_mean_shifts_m[:,any_gene_set_mask], is_dense_gene_set=full_is_dense_gene_set_m[:,any_gene_set_mask], ps=full_ps_m[:,any_gene_set_mask], sigma2s=full_sigma2s_m[:,any_gene_set_mask], return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas)

                                add_zero_mask_m = ~(__get_gene_set_mask(half_corrected_betas_mean_m, half_corrected_betas_sample_m, full_p_values_m, sparse_frac=sparse_frac_gibbs, sparse_max=sparse_max_gibbs))

                                if np.any(add_zero_mask_m):
                                    #need to convert these to the original gene sets
                                    map_to_full = np.where(any_gene_set_mask)[0]
                                    #get rows and then columns in subsetted
                                    set_to_zero_full = np.where(add_zero_mask_m)
                                    #map columns in subsetted to original
                                    set_to_zero_full = (set_to_zero_full[0], map_to_full[set_to_zero_full[1]])
                                    orig_zero = np.sum(np.any(gene_set_mask_m, axis=0))
                                    gene_set_mask_m[set_to_zero_full] = False
                                    new_zero = np.sum(np.any(gene_set_mask_m, axis=0))
                                    log("Found %d additional zero gene sets" % (orig_zero - new_zero),DEBUG)

                                    #need to update uncorrected ones too
                                    default_betas_sample_m[set_to_zero_full] = half_corrected_betas_sample_m[add_zero_mask_m]
                                    default_postp_sample_m[set_to_zero_full] = half_corrected_postp_sample_m[add_zero_mask_m]
                                    default_betas_mean_m[set_to_zero_full] = half_corrected_betas_mean_m[add_zero_mask_m]
                                    default_postp_mean_m[set_to_zero_full] = half_corrected_postp_mean_m[add_zero_mask_m]

                num_non_missing_v = np.sum(gene_set_mask_m, axis=1)
                max_num_non_missing = np.max(num_non_missing_v)
                max_num_non_missing_idx = np.argmax(num_non_missing_v)
                log("Max number of gene sets to keep across all chains is %d" % (max_num_non_missing))
                log("Keeping %d gene sets that had non-zero uncorected betas" % (sum(np.any(gene_set_mask_m, axis=0))))
                for chain_num in range(gene_set_mask_m.shape[0]):
                    if num_non_missing_v[chain_num] < max_num_non_missing:
                        cur_num = 0
                        #add in gene sets that are in the max one to ensure things are square
                        for index in np.nonzero(gene_set_mask_m[max_num_non_missing_idx,:] & ~gene_set_mask_m[chain_num,:])[0]:
                            assert(gene_set_mask_m[chain_num,index] == False)
                            gene_set_mask_m[chain_num,index] = True
                            cur_num += 1
                            if cur_num >= max_num_non_missing - num_non_missing_v[chain_num]:
                                break

                #log("Keeping %d gene sets that passed threshold of p<%.3g" % (sum(gene_set_mask), self.max_gene_set_p))


                #Now call betas in batches
                #we are doing this only for memory reasons -- we have to create a V matrix for each chain
                #it is furthermore faster to create a V once for all gene sets across all chains, and then subset it for each chain, which
                #further increases memory
                #so, the strategy is to batch the chains, for each batch calculate a V for the superset of all gene sets, and then subset it

                #betas
                if options.debug_zero_sparse:
                    full_betas_mean_m = copy.copy(default_betas_mean_m)
                    full_betas_sample_m = copy.copy(default_betas_sample_m)
                    full_postp_mean_m = copy.copy(default_postp_mean_m)
                    full_postp_sample_m = copy.copy(default_postp_sample_m)
                else:
                    full_betas_mean_m = np.zeros(default_betas_mean_m.shape)
                    full_betas_sample_m = np.zeros(default_betas_sample_m.shape)
                    full_postp_mean_m = np.zeros(default_postp_mean_m.shape)
                    full_postp_sample_m = np.zeros(default_postp_sample_m.shape)

                num_calculations = int(np.ceil(num_chains / num_batches_parallel))
                #we will default all to the uncorrected sample, and then replace those below that are non-zero
                for calc in range(num_calculations):
                    begin = calc * num_batches_parallel
                    end = (calc + 1) * num_batches_parallel
                    if end > num_chains:
                        end = num_chains

                    #get the include mask; any batch has p <= threshold
                    cur_gene_set_mask = np.any(gene_set_mask_m[begin:end,:], axis=0)
                    num_gene_set_mask = np.sum(cur_gene_set_mask)
                    max_num_gene_set_mask = np.max(np.sum(gene_set_mask_m, axis=1))

                    #construct the V matrix
                    V_superset = self._calculate_V_internal(self.X_orig[:,cur_gene_set_mask], self.y_corr_cholesky, self.mean_shifts[cur_gene_set_mask], self.scale_factors[cur_gene_set_mask])


                    #empirically it is faster to do one V if the total is less than 5x the max
                    run_one_V = num_gene_set_mask < 5 * max_num_gene_set_mask

                    if run_one_V:
                        num_non_missing = np.sum(cur_gene_set_mask)
                    else:
                        num_non_missing = np.max(np.sum(gene_set_mask_m, axis=1))

                    num_missing = gene_set_mask_m.shape[1] - num_non_missing

                    #fraction_non_missing = float(num_non_missing) / gene_set_mask_m.shape[1]
                    #missing_scale_factor = self._get_fraction_non_missing() / fraction_non_missing

                    #if missing_scale_factor > 1 / self.p:
                    #    #threshold this here. otherwise set_p will cap p but set_sigma won't cap sigma
                    #    missing_scale_factor = 1 / self.p

                    if run_one_V:
                        beta_tildes_m = full_beta_tildes_m[begin:end,cur_gene_set_mask]
                        ses_m = full_ses_m[begin:end,cur_gene_set_mask]
                        V_m=V_superset
                        scale_factors_m = self.scale_factors[cur_gene_set_mask]
                        mean_shifts_m = self.mean_shifts[cur_gene_set_mask]
                        is_dense_gene_set_m = self.is_dense_gene_set[cur_gene_set_mask]
                        ps_m = None
                        if self.ps is not None:
                            ps_m = self.ps[cur_gene_set_mask]
                        sigma2s_m = None
                        if self.sigma2s is not None:
                            sigma2s_m = self.sigma2s[cur_gene_set_mask]


                        #beta_tildes_missing_m = full_beta_tildes_m[begin:end,~cur_gene_set_mask]
                        #ses_missing_m = full_ses_m[begin:end,~cur_gene_set_mask]
                        #scale_factors_missing_m = self.scale_factors[~cur_gene_set_mask]

                    else:
                        non_missing_matrix_shape = (num_chains, num_non_missing)
                        beta_tildes_m = full_beta_tildes_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        ses_m = full_ses_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        scale_factors_m = full_scale_factors_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        mean_shifts_m = full_mean_shifts_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        is_dense_gene_set_m = full_is_dense_gene_set_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        ps_m = None
                        if full_ps_m is not None:
                            ps_m = full_ps_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        sigma2s_m = None
                        if full_sigma2s_m is not None:
                            sigma2s_m = full_sigma2s_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]

                        V_m = np.zeros((end-begin, beta_tildes_m.shape[1], beta_tildes_m.shape[1]))
                        for i,j in zip(range(begin, end),range(end-begin)):
                            #gene_set_mask_m[i,:] is the current batch mask, with dimensions num_gene_sets
                            #to index into V_superset, we need to subset it down to cur_gene_set_mask
                            gene_set_mask_subset = gene_set_mask_m[i,cur_gene_set_mask]
                            V_m[j,:,:] = V_superset[gene_set_mask_subset,:][:,gene_set_mask_subset]

                        #missing_matrix_shape = (num_chains, num_missing)
                        #beta_tildes_missing_m = full_beta_tildes_m[~gene_set_mask_m].reshape(missing_matrix_shape)[begin:end,:]
                        #ses_missing_m = full_ses_m[~gene_set_mask_m].reshape(missing_matrix_shape)[begin:end,:]
                        #scale_factors_missing_m = full_scale_factors_m[~gene_set_mask_m].reshape(missing_matrix_shape)[begin:end,:]

                    (cur_betas_sample_m, cur_postp_sample_m, cur_betas_mean_m, cur_postp_mean_m) = self._calculate_non_inf_betas(initial_p=None, beta_tildes=beta_tildes_m, ses=ses_m, V=V_m, scale_factors=scale_factors_m, mean_shifts=mean_shifts_m, is_dense_gene_set=is_dense_gene_set_m, ps=ps_m, sigma2s=sigma2s_m, return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, num_missing_gene_sets=num_missing, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, debug_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if gene_set_mask_m[0,i]])

                    #store the values with zeros appended in order to add to sum_betas_m below
                    if run_one_V:
                        full_betas_sample_m[begin:end,cur_gene_set_mask] = cur_betas_sample_m
                        full_postp_sample_m[begin:end,cur_gene_set_mask] = cur_postp_sample_m
                        full_betas_mean_m[begin:end,cur_gene_set_mask] = cur_betas_mean_m
                        full_postp_mean_m[begin:end,cur_gene_set_mask] = cur_postp_mean_m

                        #handy option for debugging
                        print_overlapping = None
                        if print_overlapping is not None:
                            gene_sets_run = [self.gene_sets[i] for i in range(len(self.gene_sets)) if cur_gene_set_mask[i]]
                            gene_set_to_ind = self._construct_map_to_ind(gene_sets_run)
                            for gene_set in print_overlapping:
                                if gene_set in gene_set_to_ind:
                                    log("For gene set %s" % (gene_set))
                                    ind = gene_set_to_ind[gene_set]
                                    values = V_m[ind,:] * (cur_betas_mean_m if use_mean_betas else cur_betas_sample_m)
                                    indices = np.argsort(values, axis=1)
                                    for chain in range(values.shape[0]):
                                        log("Chain %d (uncorrected beta=%.4g, corrected beta=%.4g)" % (chain, uncorrected_betas_mean_m[chain,self.gene_set_to_ind[gene_set]], (cur_betas_mean_m[chain,ind] if use_mean_betas else cur_betas_sample_m[chain,ind])))
                                        for i in indices[chain,::-1]:
                                            if values[chain,i] == 0:
                                                break
                                            log("%s, V=%.4g, beta=%.4g, prod=%.4g" % (gene_sets_run[i], V_m[ind,i], (cur_betas_mean_m[chain,i] if use_mean_betas else cur_betas_sample_m[chain,i]), values[chain,i]))

                    else:
                        #store the values with zeros appended in order to add to sum_betas_m below
                        full_betas_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_sample_m.ravel()
                        full_postp_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_sample_m.ravel()
                        full_betas_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_mean_m.ravel()
                        full_postp_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_mean_m.ravel()

                    #see how many are set to zero
                    #set_to_zero_v += np.mean(np.logical_and(full_betas_sample_m == 0, uncorrected_betas_sample_m == 0).reshape(full_betas_sample_m.shape), axis=0)
                    #avg_full_betas_sample_v += np.mean(full_betas_sample_m, axis=0)
                    #avg_full_postp_sample_v += np.mean(full_postp_sample_m, axis=0)

                #now restore the p and sigma
                #self.set_sigma(orig_sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
                #self.set_p(orig_p)

                #since the betas are a sample (rather than mean), we can sample from priors by just multiplying this sample

                #this is the (log) prior odds relative to the background_log_bf

                def __calc_priors(_X, _betas, mean_shifts, scale_factors):
                    return np.array(_X.dot((_betas / scale_factors).T) - np.sum(mean_shifts * _betas / scale_factors, axis=1).T).T


                priors_sample_m = __calc_priors(self.X_orig, full_betas_sample_m, self.mean_shifts, self.scale_factors)
                priors_mean_m = __calc_priors(self.X_orig, full_betas_mean_m, self.mean_shifts, self.scale_factors)
                if self.genes_missing is not None:
                    priors_missing_sample_m = __calc_priors(self.X_orig_missing_genes, full_betas_sample_m, self.mean_shifts, self.scale_factors)
                    priors_missing_mean_m = __calc_priors(self.X_orig_missing_genes, full_betas_mean_m, self.mean_shifts, self.scale_factors)

                def __adjust_max_prior_contribution(_X, _betas, _priors_m):
                    priors_max_contribution_m = np.zeros(_priors_m.shape)
                    #don't think it is possible to vectorize this due to sparse matrices maxxing out at two dimensions
                    for chain in range(priors_max_contribution_m.shape[0]):
                        priors_max_contribution_m[chain,:] = _X.multiply(np.abs(_betas[chain,:]) / self.scale_factors).max(axis=1).todense().A1
                    priors_naive_m = np.array(_X.dot((np.abs(_betas) / self.scale_factors).T)).T

                    priors_percentage_max_m = np.ones(_priors_m.shape)
                    non_zero_priors_mask = priors_naive_m != 0
                    priors_percentage_max_m[non_zero_priors_mask] = priors_max_contribution_m[non_zero_priors_mask] / priors_naive_m[non_zero_priors_mask]

                    priors_percentage_max_m[priors_percentage_max_m < 0] = 0
                    priors_percentage_max_m[priors_percentage_max_m > 1] = 1

                    #for each prior, sample one percentage
                    new_priors_percentage_max_m = copy.copy(priors_percentage_max_m)
                    max_allowed_percentage = 0.95
                    for chain in range(priors_max_contribution_m.shape[0]):
                        sample_mask = priors_percentage_max_m[chain,:] < max_allowed_percentage
                        num_allowed = np.sum(sample_mask)
                        if num_allowed > 0:
                            new_columns = np.random.randint(num_allowed, size=_priors_m.shape[1])
                            new_priors_percentage_max_m[chain,:] = priors_percentage_max_m[chain,sample_mask][new_columns]

                    #don't update those that were below the threshold or would increase
                    no_update_priors_mask = np.logical_or(priors_percentage_max_m < new_priors_percentage_max_m, priors_percentage_max_m < max_allowed_percentage)
                    new_priors_percentage_max_m[no_update_priors_mask] = priors_percentage_max_m[no_update_priors_mask]

                    #return (np.zeros(_priors_m.shape), priors_percentage_max_m)

                    priors_adjustment_m = -priors_max_contribution_m + new_priors_percentage_max_m * priors_naive_m
                    return (priors_adjustment_m, priors_percentage_max_m)

                #option not used right now
                #if see one gene set dominating top, consider adding back
                penalize_priors = False
                if penalize_priors:
                    self._record_param("penalize_priors", True)

                    #first calculate
                    priors_to_adjust_sample_m = priors_sample_m
                    priors_to_adjust_mean_m = priors_mean_m
                    if self.genes_missing is not None:
                        priors_to_adjust_sample_m = np.hstack((priors_sample_m, priors_missing_sample_m))
                        priors_to_adjust_mean_m = np.hstack((priors_mean_m, priors_missing_mean_m))

                    (priors_adjustment_sample_m, priors_percentage_max_sample_m) = __adjust_max_prior_contribution(X_all, full_betas_sample_m, priors_to_adjust_sample_m)
                    (priors_adjustment_mean_m, priors_percentage_max_mean_m) = __adjust_max_prior_contribution(X_all, full_betas_mean_m, priors_to_adjust_mean_m)

                    if self.genes_missing is not None:
                        priors_missing_sample_m += priors_adjustment_sample_m[:,-priors_missing_sample_m.shape[1]:]
                        priors_missing_mean_m += priors_adjustment_mean_m[:,-priors_missing_mean_m.shape[1]:]
                        priors_sample_m += priors_adjustment_sample_m[:,:priors_adjustment_sample_m.shape[1]-priors_missing_sample_m.shape[1]]
                        priors_mean_m += priors_adjustment_mean_m[:,:priors_adjustment_mean_m.shape[1]-priors_missing_mean_m.shape[1]]
                    else:
                        priors_sample_m += priors_adjustment_sample_m
                        priors_mean_m += priors_adjustment_mean_m

                if self.huge_signal_bfs is not None and update_huge_scores:
                    #Now update the BFs is we have huge scores
                    log("Updating HuGE scores")
                    if self.huge_signal_bfs is not None:
                        rel_prior_log_bf = priors_for_Y_m

                        (log_bf_m, log_bf_uncorrected_m, absent_genes, absent_log_bf) = self._distill_huge_signal_bfs(self.huge_signal_bfs_for_regression, self.huge_signal_posteriors_for_regression, self.huge_signal_sum_gene_cond_probabilities_for_regression, self.huge_signal_mean_gene_pos_for_regression, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, self.gene_covariates, self.gene_covariates_mask, self.gene_covariates_mat_inv, self.gene_covariate_names, self.gene_covariate_intercept_index, self.genes, rel_prior_log_bf=rel_prior_log_bf + (self.Y_exomes if self.Y_exomes is not None else 0) + (self.Y_positive_controls if self.Y_positive_controls is not None else 0))

                        if compute_Y_raw:
                            (log_bf_raw_m, log_bf_uncorrected_raw_m, absent_genes_raw, absent_log_bf_raw) = self._distill_huge_signal_bfs(self.huge_signal_bfs, self.huge_signal_posteriors, self.huge_signal_sum_gene_cond_probabilities, self.huge_signal_mean_gene_pos, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, self.gene_covariates, self.gene_covariates_mask, self.gene_covariates_mat_inv, self.gene_covariate_names, self.gene_covariate_intercept_index, self.genes, rel_prior_log_bf=rel_prior_log_bf + (self.Y_exomes if self.Y_exomes is not None else 0) + (self.Y_positive_controls if self.Y_positive_controls is not None else 0))
                        else:
                            log_bf_raw_m = copy.copy(log_bf_m)

                        if self.Y_exomes is not None:
                            #add in the Y_exomes
                            #it was used in distill just to fine map the GWAS associations; it was then subtracted out
                            #the other component of the rel_prior_log_bf (priors) will be added back in next iteration
                            log_bf_m += self.Y_exomes
                            log_bf_uncorrected_m += self.Y_exomes
                            log_bf_raw_m += self.Y_exomes

                        if self.Y_positive_controls is not None:
                            #add in the Y_positive_controls
                            #it was used in distill just to fine map the GWAS associations; it was then subtracted out
                            #the other component of the rel_prior_log_bf (priors) will be added back in next iteration
                            log_bf_m += self.Y_positive_controls
                            log_bf_uncorrected_m += self.Y_positive_controls
                            log_bf_raw_m += self.Y_positive_controls

                        if len(absent_genes) > 0:
                            bail("Error: huge_signal_bfs was incorrectly set and contains extra genes")

                #if center_combined:
                #    priors_sample_total_m = np.hstack((priors_sample_m, priors_missing_sample_m))
                #    total_mean_v = np.mean(priors_sample_total_m, axis=1)
                #    priors_sample_m = (priors_sample_m.T - total_mean_v).T
                #    priors_missing_sample_m = (priors_missing_sample_m.T - total_mean_v).T
                #    priors_mean_m = (priors_mean_m.T - total_mean_v).T
                #    priors_missing_mean_m = (priors_missing_mean_m.T - total_mean_v).T


                #    priors_sample_m = (priors_sample_m.T + np.mean(full_alpha_tildes_m, axis=1) - self.background_log_bf).T
                #    priors_missing_sample_m = (priors_missing_sample_m.T + np.mean(full_alpha_tildes_m, axis=1) - self.background_log_bf).T

                #do the regression
                total_priors_m = np.hstack((priors_sample_m, priors_missing_sample_m))
                gene_N = self.get_gene_N()
                gene_N_missing = self.get_gene_N(get_missing=True)

                all_gene_N = gene_N
                if self.genes_missing is not None:
                    assert(gene_N_missing is not None)
                    all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

                priors_slope = total_priors_m.dot(all_gene_N) / (total_priors_m.shape[1] * np.var(all_gene_N))
                #no intercept since we just standardized above

                if adjust_priors:
                    log("Adjusting priors with slopes ranging from %.4g-%.4g" % (np.min(priors_slope), np.max(priors_slope)), TRACE)
                    priors_sample_m = priors_sample_m - np.outer(priors_slope, gene_N)
                    priors_mean_m = priors_mean_m - np.outer(priors_slope, gene_N)

                    if self.genes_missing is not None:
                        priors_missing_sample_m = priors_missing_sample_m - np.outer(priors_slope, gene_N_missing)
                        priors_missing_mean_m = priors_missing_mean_m - np.outer(priors_slope, gene_N_missing)

                priors_for_Y_m = priors_sample_m
                priors_percentage_max_for_Y_m = priors_percentage_max_sample_m
                priors_adjustment_for_Y_m = priors_adjustment_sample_m
                if use_mean_betas:
                    priors_for_Y_m = priors_mean_m                
                    priors_percentage_max_for_Y_m = priors_percentage_max_mean_m
                    priors_adjustment_for_Y_m = priors_adjustment_mean_m

                #only add non-outliers to mean/sd
                #non_outlier_mask = np.full(sum_z_scores2_m.shape, True)
                #non_outlier_mask[num_sum_outlier_m > 10] = np.abs(full_z_scores_m[num_sum_outlier_m > 10]) < -scipy.stats.norm.ppf(0.5 * 0.05 / (num_sum_outlier_m[num_sum_outlier_m > 10] * num_chains_betas))
                #if pre_gene_set_filter_mask is not None:
                #    non_outlier_mask[:,~pre_gene_set_filter_mask] = False
                #sum_z_scores2_m[non_outlier_mask] = np.add(sum_z_scores2_m[non_outlier_mask], np.power(full_z_scores_m[non_outlier_mask], 2))
                #sum_z_scores_m[non_outlier_mask] = np.add(sum_z_scores_m[non_outlier_mask], full_z_scores_m[non_outlier_mask])
                #num_sum_outlier_m[non_outlier_mask] += 1

                all_sum_betas_m = np.add(all_sum_betas_m, full_betas_mean_m)
                all_sum_betas2_m = np.add(all_sum_betas2_m, np.power(full_betas_mean_m, 2))
                all_sum_z_scores_m = np.add(all_sum_z_scores_m, full_z_scores_m)
                all_sum_z_scores2_m = np.add(all_sum_z_scores2_m, np.power(full_z_scores_m, 2))
                all_num_sum_m += 1

                all_sum_Ys_m = np.add(all_sum_Ys_m, Y_sample_m)
                all_sum_Ys2_m = np.add(all_sum_Ys2_m, np.power(Y_sample_m, 2))

                R_Y_v = np.zeros(all_sum_Ys_m.shape[1])
                R_beta_v = np.zeros(all_sum_betas_m.shape[1])

                if increase_hyper_if_betas_below is not None:
                    #check to make sure that we satisfy
                    if np.any(all_num_sum_m == 0):
                        gibbs_good = False
                    else:

                        #check both sum of all iterations (to not wait until convergence to detect failures)
                        #and sum of iterations after convergence
                        _, all_cur_avg_betas_v = __outlier_resistant_mean(all_sum_betas_m, all_num_sum_m)

                        fraction_required = 0.001
                        self._record_param("fraction_required_to_not_increase_hyper", fraction_required)

                        #all_low = np.all(all_cur_avg_betas_v / self.scale_factors < increase_hyper_if_betas_below)
                        #all_low = np.mean(all_cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below) < fraction_required
                        all_low = False

                        if np.all(num_sum_beta_m > 0):
                            _, cur_avg_betas_v = __outlier_resistant_mean(sum_betas_m, num_sum_beta_m)
                            #all_low = all_low or np.all(cur_avg_betas_v / self.scale_factors < increase_hyper_if_betas_below)
                            #all_low = np.all(cur_avg_betas_v / self.scale_factors < increase_hyper_if_betas_below)
                            all_low = np.mean(cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below) < fraction_required
                            
                        #if np.all(all_num_sum_m > 0):
                        #    top_gene_set = np.argmax(np.mean(all_sum_betas_m / all_num_sum_m, axis=0) / self.scale_factors)
                        #    log("Top gene set %s has all value %.3g" % (self.gene_sets[top_gene_set], (np.mean(all_sum_betas_m / all_num_sum_m, axis=0) / self.scale_factors)[top_gene_set]), TRACE)
                        #    top_gene_set2 = np.argmax(all_cur_avg_betas_v / self.scale_factors)
                        #    log("Top gene set %s has all outlier value %.3g" % (self.gene_sets[top_gene_set], (all_cur_avg_betas_v / self.scale_factors)[top_gene_set]), TRACE)

                        if np.all(num_sum_beta_m > 0):
                            top_gene_set = np.argmax(np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)
                            log("Top gene set %s has value %.3g" % (self.gene_sets[top_gene_set], (np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)[top_gene_set]), TRACE)
                            top_gene_set2 = np.argmax(cur_avg_betas_v / self.scale_factors)
                            log("Top gene set %s has outlier value %.3g" % (self.gene_sets[top_gene_set2], (cur_avg_betas_v / self.scale_factors)[top_gene_set]), TRACE)

                            #top_gene_sets = np.argsort(np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)[::-1][:10]
                            #for i in top_gene_sets:
                            #    log("Top %d gene set %s has value %.3g" % (i, self.gene_sets[i], (np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)[i]), TRACE)

                            #top_gene_sets = np.argsort(cur_avg_betas_v / self.scale_factors)[::-1][:10]
                            #for i in top_gene_sets:
                            #    log("Top %d gene set %s has outlier value %.3g" % (i, self.gene_sets[i], (cur_avg_betas_v[i] / self.scale_factors)[i]), TRACE)


                        if all_low:

                            log("Only %.3g of %d (%.3g) are above %.3g" % (np.sum(cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below), len(cur_avg_betas_v), np.mean(cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below), increase_hyper_if_betas_below))

                            #at minimum, guarantee that it will restart unless it gets above this
                            gibbs_good = False
                            #only if above num for checking though that we increase and restart
                            if iteration_num > num_before_checking_p_increase:
                                new_p = self.p
                                new_sigma2 = self.sigma2

                                self._record_param("p_scale_factor", p_scale_factor)

                                new_p = self.p * p_scale_factor
                                num_p_increases += 1
                                if new_p > 1:
                                    new_p = 1

                              
                                break_loop = False
                                if new_p != self.p and num_restarts < max_num_restarts:
                                    #update so that new_sigma2 / new_p = self.sigma2 / self.p
                                    new_sigma2 = self.sigma2 * new_p / self.p

                                    self.ps *= new_p / self.p
                                    self.set_p(new_p)
                                    self._record_param("p_adj", new_p)
                                    log("Detected all gene set betas below %.3g; increasing p to %.3g and restarting gibbs" % (increase_hyper_if_betas_below, self.p))

                                    #restart
                                    break_loop = True
                                if new_sigma2 != self.sigma2 and num_restarts < max_num_restarts:
                                    self.sigma2s *= new_sigma2 / self.sigma2
                                    self._record_param("sigma2_adj", new_sigma2)
                                    self.set_sigma(new_sigma2, self.sigma_power)
                                    log("Detected all gene set betas below %.3g; increasing sigma to %.3g and restarting gibbs" % (increase_hyper_if_betas_below, self.sigma2))
                                    break_lopp = True
                                if break_loop:
                                    break
                        else:
                            gibbs_good = True

                if np.any(np.concatenate((burn_in_phase_Y_v, burn_in_phase_beta_v))):
                    if iteration_num + 1 >= max_num_burn_in:
                        burn_in_phase_Y_v[:] = False
                        burn_in_phase_beta_v[:] = False
                        log("Stopping Gibbs burn in after %d iterations" % (iteration_num+1), INFO)
                    elif gauss_seidel:
                        if prev_Ys_m is not None:
                            sum_diff = np.sum(np.abs(prev_Ys_m - Y_sample_m))
                            sum_prev = np.sum(np.abs(prev_Ys_m))
                            max_diff_frac = np.max(np.abs((prev_Ys_m - Y_sample_m)/prev_Ys_m))

                            tot_diff = sum_diff / sum_prev
                            log("Gibbs iteration %d: mean gauss seidel difference = %.4g / %.4g = %.4g; max frac difference = %.4g" % (iteration_num+1, sum_diff, sum_prev, tot_diff, max_diff_frac))
                            if iteration_num > min_num_iter and tot_diff < eps:
                                log("Gibbs gauss converged after %d iterations" % iteration_num, INFO)
                                burn_in_phase_Y_v[:] = False
                                burn_in_phase_beta_v[:] = False

                        prev_Ys_m = Y_sample_m
                    else:
                        def __calculate_R(sum_m, sum2_m, num):
                            #mean of betas across all iterations; psi_dot_j
                            mean_m = sum_m / float(num)
                            #mean of betas across replicates; psi_dot_dot
                            mean_v = np.mean(mean_m, axis=0)
                            #variances of betas across all iterators; s_j
                            var_m = (sum2_m - float(num) * np.power(mean_m, 2)) / (float(num) - 1)
                            B_v = (float(num) / (mean_m.shape[0] - 1)) * np.sum(np.power(mean_m - mean_v, 2), axis=0)
                            W_v = (1.0 / float(mean_m.shape[0])) * np.sum(var_m, axis=0)
                            var_given_y_v = np.add((float(num) - 1) / float(num) * W_v, (1.0 / float(num)) * B_v)
                            var_given_y_v[var_given_y_v < 0] = 0
                            R_v = np.ones(len(W_v))
                            R_non_zero_mask = W_v > 0
                            R_v[R_non_zero_mask] = np.sqrt(var_given_y_v[R_non_zero_mask] / W_v[R_non_zero_mask])
                            return (B_v, W_v, R_v, var_given_y_v)

                        if iteration_num > min_num_burn_in:
                            (B_Y_v, W_Y_v, R_Y_v, var_given_y_Y_v) = __calculate_R(all_sum_Ys_m, all_sum_Ys2_m, iteration_num)
                            (B_beta_v, W_beta_v, R_beta_v, var_given_y_beta_v) = __calculate_R(all_sum_betas_m, all_sum_betas2_m, iteration_num)

                            B_v = np.concatenate((B_Y_v, B_beta_v))
                            W_v = np.concatenate((W_Y_v, W_beta_v))
                            R_v = np.concatenate((R_Y_v, R_beta_v))
                            W_v = np.concatenate((R_Y_v, R_beta_v))

                            mean_thresholded_R_Y = np.mean(R_Y_v[R_Y_v >= 1]) if np.sum(R_Y_v >= 1) > 0 else 1
                            max_index_Y = np.argmax(R_Y_v)
                            mean_thresholded_R_beta = np.mean(R_beta_v[R_beta_v >= 1]) if np.sum(R_beta_v >= 1) > 0 else 1
                            max_index_beta = np.argmax(R_beta_v)
                            mean_thresholded_R = np.mean(R_v[R_v >= 1]) if np.sum(R_v >= 1) > 0 else 1
                            max_index = np.argmax(R_v)
                            log("Gibbs iteration %d: max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (iteration_num+1, max_index, B_v[max_index], W_v[max_index], R_v[max_index], mean_thresholded_R, np.sum(R_v > r_threshold_burn_in)))
                            log("For Y: max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (max_index_Y, B_Y_v[max_index_Y], W_Y_v[max_index_Y], R_Y_v[max_index_Y], mean_thresholded_R_Y, np.sum(R_Y_v > r_threshold_burn_in)), TRACE)
                            log("For beta: max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (max_index_beta, B_beta_v[max_index_beta], W_beta_v[max_index_beta], R_beta_v[max_index_beta], mean_thresholded_R_beta, np.sum(R_beta_v > r_threshold_burn_in)), TRACE)

                            if use_max_r_for_convergence:
                                convergence_statistic = R_v[max_index]
                            else:
                                convergence_statistic = mean_thresholded_R


                            num_Y_converged = np.sum(np.logical_and(burn_in_phase_Y_v, R_Y_v < r_threshold_burn_in))
                            num_beta_converged = np.sum(np.logical_and(burn_in_phase_beta_v, R_beta_v < r_threshold_burn_in))

                            burn_in_phase_Y_v[R_Y_v < r_threshold_burn_in] = False
                            burn_in_phase_beta_v[R_beta_v < r_threshold_burn_in] = False

                            if num_Y_converged > 0:
                                log("Gibbs converged for %d Ys (%d remaining) after %d iterations" % (num_Y_converged, np.sum(burn_in_phase_Y_v), iteration_num+1), INFO)
                            if num_beta_converged > 0:
                                log("Gibbs converged for %d betas (%d remaining) after %d iterations" % (num_beta_converged, np.sum(burn_in_phase_beta_v), iteration_num+1), INFO)

                done = False

                betas_sem2_v = np.zeros(sum_betas_m.shape[1])
                sem2_v = np.zeros(sum_log_pos_m.shape[1])

                converged_Y_v = ~burn_in_phase_Y_v
                converged_beta_v = ~burn_in_phase_beta_v

                if np.sum(converged_Y_v) + np.sum(converged_beta_v) > 0:

                    #sum_Ys_post_m = np.add(sum_Ys_post_m, Y_sample_m)
                    #sum_Ys2_post_m += np.add(sum_Ys2_post_m, np.square(Y_sample_m))
                    #num_sum_post += 1

                    if np.sum(converged_Y_v) > 0:
                        sum_Ys_m[:,converged_Y_v] += Y_sample_m[:,converged_Y_v]
                        sum_Ys2_m[:,converged_Y_v] += np.power(Y_sample_m[:,converged_Y_v], 2)
                        sum_Y_raws_m[:,converged_Y_v] += Y_raw_sample_m[:,converged_Y_v]
                        sum_log_pos_m[:,converged_Y_v] += log_po_sample_m[:,converged_Y_v]
                        sum_log_pos2_m[:,converged_Y_v] += np.power(log_po_sample_m[:,converged_Y_v], 2)
                        sum_log_po_raws_m[:,converged_Y_v] += log_po_raw_sample_m[:,converged_Y_v]
                        sum_priors_m[:,converged_Y_v] += priors_for_Y_m[:,converged_Y_v]
                        sum_Ds_m[:,converged_Y_v] += D_sample_m[:,converged_Y_v]
                        sum_D_raws_m[:,converged_Y_v] += D_raw_sample_m[:,converged_Y_v]
                        sum_bf_orig_m[:,converged_Y_v] += log_bf_m[:,converged_Y_v]
                        sum_bf_uncorrected_m[:,converged_Y_v] += log_bf_uncorrected_m[:,converged_Y_v]
                        sum_bf_orig_raw_m[:,converged_Y_v] += log_bf_raw_m[:,converged_Y_v]
                        num_sum_Y_m[:,converged_Y_v] += 1


                    #temp_genes = ["FTO", "IRS1", "ANKH", "INSR"]
                    #temp_genes = [x for x in temp_genes if x in self.gene_to_ind]

                    if np.sum(converged_beta_v) > 0:

                        sum_betas_m[:,converged_beta_v] += full_betas_mean_m[:,converged_beta_v]
                        sum_betas2_m[:,converged_beta_v] += np.power(full_betas_mean_m[:,converged_beta_v], 2)
                        sum_betas_uncorrected_m[:,converged_beta_v] += uncorrected_betas_mean_m[:,converged_beta_v]
                        sum_postp_m[:,converged_beta_v] += full_postp_sample_m[:,converged_beta_v]
                        sum_beta_tildes_m[:,converged_beta_v] += full_beta_tildes_m[:,converged_beta_v]
                        sum_z_scores_m[:,converged_beta_v] += full_z_scores_m[:,converged_beta_v]
                        num_sum_beta_m[:,converged_beta_v] += 1

                    if self.genes_missing is not None and np.sum(burn_in_phase_beta_v) == 0:
                        burn_in_phase_Y_missing_v = (self.X_orig_missing_genes != 0).multiply(burn_in_phase_beta_v).sum(axis=1).astype(bool).A1

                        converged_Y_missing_v = ~burn_in_phase_Y_missing_v

                        if np.sum(converged_Y_missing_v) > 0:

                            sum_priors_missing_m[:,converged_Y_missing_v] += priors_missing_mean_m[:,converged_Y_missing_v]

                            max_log = 15
                            cur_log_priors_missing_m = priors_missing_mean_m[:,converged_Y_missing_v] + self.background_log_bf
                            cur_log_priors_missing_m[cur_log_priors_missing_m > max_log] = max_log

                            sum_Ds_missing_m[:,converged_Y_missing_v] += np.exp(cur_log_priors_missing_m) / (1 + np.exp(cur_log_priors_missing_m))

                            num_sum_priors_missing_m[:,converged_Y_missing_v] += 1


                    #record these for tracing

                    if np.all(num_sum_Y_m > 1) and np.all(num_sum_beta_m > 1):
                        betas_sem2_m = ((sum_betas2_m / (num_sum_beta_m - 1)) - np.power(sum_betas_m / num_sum_beta_m, 2)) / num_sum_beta_m

                        #calculate effective sample size
                        #see: https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
                        #we have to keep rho_t for t=1...2m+1
                        #to get rho_t, multiply current Y_sample_m

                        #first the correlation vectors
                        #rho_t = np.zeros(len(var_given_y_v))
                        #for i in range(num_chains_betas):
                        #    np.correlate(
                        #    rho_t += 

                        #1 - ((W_v - (1 / num_chains_betas) * sum_rho_t_m) / var_given_y_v)

                        avg_v = np.mean(sum_log_pos_m / num_sum_Y_m, axis=0)
                        sem2_v = np.var(sum_log_pos_m / num_sum_Y_m, axis=0) / np.mean(num_sum_Y_m, axis=0)

                        max_avg = np.max(avg_v)
                        min_avg = np.min(avg_v)
                        ref_val = max_avg - min_avg
                        if ref_val == 0:
                            ref_val = np.sqrt(np.var(Y_sample_m))
                            if ref_val == 0:
                                ref_val = 1

                        max_sem = np.max(np.sqrt(sem2_v))
                        max_percentage_error = max_sem / ref_val

                        log("Gibbs iteration %d: ref_val=%.3g; max_sem=%.3g; max_ratio=%.3g" % (iteration_num+1, ref_val, max_sem, max_percentage_error))
                        if iteration_num >= min_num_iter and max_percentage_error < max_frac_sem and (increase_hyper_if_betas_below is not None and iteration_num >= num_before_checking_p_increase):
                            log("Desired Gibbs precision achieved; stopping sampling")
                            done = True

                if gene_set_stats_trace_out is not None:
                    log("Writing gene set stats trace", TRACE)
                    for chain_num in range(num_chains):
                        for i in range(len(self.gene_sets)):
                            if only_gene_sets is None or self.gene_sets[i] in only_gene_sets:
                                gene_set_stats_trace_fh.write("%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num+1, chain_num+1, self.gene_sets[i], full_beta_tildes_m[chain_num,i] / self.scale_factors[i], full_p_values_m[chain_num,i], full_z_scores_m[chain_num,i], full_ses_m[chain_num,i] / self.scale_factors[i], (uncorrected_betas_mean_m[chain_num,i] if use_mean_betas else uncorrected_betas_sample_m[chain_num,i])  / self.scale_factors[i], (full_betas_mean_m[chain_num,i] if use_mean_betas else full_betas_sample_m[chain_num,i]) / self.scale_factors[i], (full_postp_mean_m[chain_num,i] if use_mean_betas else full_postp_sample_m[chain_num,i]), full_z_cur_beta_tildes_m[chain_num,i], R_beta_v[i], betas_sem2_v[i]))

                    gene_set_stats_trace_fh.flush()

                if done:
                    break

            if gene_set_stats_trace_out is not None:
                gene_set_stats_trace_fh.close()
            if gene_stats_trace_out is not None:
                gene_stats_trace_fh.close()

            #reached the max; go with whatever we have
            if num_restarts >= max_num_restarts:
                gibbs_good = True

            #restart if not good
            if not gibbs_good:
                continue

            assert(np.all(num_sum_Y_m > 0))
            assert(np.all(num_sum_beta_m > 0))

            ##1. calculate mean values for each chain (divide by number -- make sure it is correct; may not be num_avg_Y)
            #beta_chain_means_m = sum_betas_m / num_sum_beta_m
            #Y_chain_means_m = sum_Ys_m / num_sum_Y_m

            ##2. calculate median values across chains (one number per gene set/gene)
            #beta_medians_v = np.median(beta_chain_means_m, axis=0)
            #Y_medians_v = np.median(Y_chain_means_m, axis=0)

            ##3. calculate abs(difference) between each chain and median (one value per chain/geneset)
            #beta_mad_m = np.abs(beta_chain_means_m - beta_medians_v)
            #Y_mad_m = np.abs(Y_chain_means_m - Y_medians_v)

            ##4. calculate median of abs(difference) across chains (one number per gene set/gene)
            #beta_mad_median_v = np.median(beta_mad_m, axis=0)
            #Y_mad_median_v = np.median(Y_mad_m, axis=0)

            ##5. mask any chain that is more than 3 median(abs(difference)) from median
            #beta_outlier_mask_m = beta_chain_means_m > beta_medians_v + 3 * beta_mad_median_v
            #Y_outlier_mask_m = Y_chain_means_m > Y_medians_v + 3 * Y_mad_median_v

            ##6. take average only across chains that are not outliers
            #num_sum_beta_v = np.sum(~beta_outlier_mask_m, axis=0)
            #num_sum_Y_v = np.sum(~Y_outlier_mask_m, axis=0)

            ##should never happen but just in case
            #num_sum_beta_v[num_sum_beta_v == 0] = 1
            #num_sum_Y_v[num_sum_Y_v == 0] = 1

            ##7. to do this, zero out outlier chains, then sum them, then divide by number of outliers
            #sum_Ys_m[Y_outlier_mask_m] = 0
            #avg_Ys_v = np.sum(sum_Ys_m / num_sum_Y_m, axis=0) / num_sum_Y_v

            Y_outlier_mask_m, avg_Ys_v = __outlier_resistant_mean(sum_Ys_m, num_sum_Y_m)
            beta_outlier_mask_m, avg_betas_v = __outlier_resistant_mean(sum_betas_m, num_sum_beta_m)
            
            _, avg_Y_raws_v = __outlier_resistant_mean(sum_Y_raws_m, num_sum_Y_m)

            #sum_log_pos_m[Y_outlier_mask_m] = 0
            #avg_log_pos_v = np.sum(sum_log_pos_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_log_pos_v = __outlier_resistant_mean(sum_log_pos_m, num_sum_Y_m, Y_outlier_mask_m)

            _, avg_log_po_raws_v = __outlier_resistant_mean(sum_log_po_raws_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_log_pos2_m[Y_outlier_mask_m] = 0
            #avg_log_pos2_v = np.sum(sum_log_pos2_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_log_pos2_v = __outlier_resistant_mean(sum_log_pos2_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_Ds_m[Y_outlier_mask_m] = 0
            #avg_Ds_v = np.sum(sum_Ds_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_Ds_v = __outlier_resistant_mean(sum_Ds_m, num_sum_Y_m, Y_outlier_mask_m)

            _, avg_D_raws_v = __outlier_resistant_mean(sum_D_raws_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_priors_m[Y_outlier_mask_m] = 0
            #avg_priors_v = np.sum(sum_priors_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_priors_v = __outlier_resistant_mean(sum_priors_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_bf_orig_m[Y_outlier_mask_m] = 0
            #avg_bf_orig_v = np.sum(sum_bf_orig_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_bf_orig_v = __outlier_resistant_mean(sum_bf_orig_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_bf_orig_raw_m[Y_outlier_mask_m] = 0
            #avg_bf_orig_raw_v = np.sum(sum_bf_orig_raw_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_bf_orig_raw_v = __outlier_resistant_mean(sum_bf_orig_raw_m, num_sum_Y_m, Y_outlier_mask_m)

            if self.genes_missing is not None:
                #priors_missing_chain_means_m = sum_priors_missing_m / num_sum_priors_missing_m
                #priors_missing_medians_v = np.median(priors_missing_chain_means_m, axis=0)
                #priors_missing_mad_m = np.abs(priors_missing_chain_means_m - priors_missing_medians_v)
                #priors_missing_mad_median_v = np.median(priors_missing_mad_m, axis=0)
                #priors_missing_outlier_mask_m = priors_missing_chain_means_m > priors_missing_medians_v + 3 * priors_missing_mad_median_v
                #num_sum_priors_missing_v = np.sum(~priors_missing_outlier_mask_m, axis=0)
                #num_sum_priors_missing_v[num_sum_priors_missing_v == 0] = 1

                #assert(np.all(num_sum_priors_missing_m > 0))
                #sum_priors_missing_m[priors_missing_outlier_mask_m] = 0
                #avg_priors_missing_v = np.sum(sum_priors_missing_m / num_sum_priors_missing_m, axis=0) / num_sum_priors_missing_v

                priors_missing_outlier_mask_m, avg_priors_missing_v = __outlier_resistant_mean(sum_priors_missing_m, num_sum_priors_missing_m)

                #sum_Ds_missing_m[priors_missing_outlier_mask_m] = 0
                #avg_Ds_missing_v = np.sum(sum_Ds_missing_m / num_sum_priors_missing_m, axis=0) / num_sum_priors_missing_v
                _, avg_Ds_missing_v = __outlier_resistant_mean(sum_Ds_missing_m, num_sum_priors_missing_m, priors_missing_outlier_mask_m)

            #sum_betas_m[beta_outlier_mask_m] = 0
            #avg_betas_v = np.sum(sum_betas_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            #we did this above

            #sum_betas_uncorrected_m[beta_outlier_mask_m] = 0
            #avg_betas_uncorrected_v = np.sum(sum_betas_uncorrected_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_betas_uncorrected_v = __outlier_resistant_mean(sum_betas_uncorrected_m, num_sum_beta_m, beta_outlier_mask_m)

            #sum_postp_m[beta_outlier_mask_m] = 0
            #avg_postp_v = np.sum(sum_postp_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_postp_v = __outlier_resistant_mean(sum_postp_m, num_sum_beta_m, beta_outlier_mask_m)

            #sum_beta_tildes_m[beta_outlier_mask_m] = 0
            #avg_beta_tildes_v = np.sum(sum_beta_tildes_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_beta_tildes_v = __outlier_resistant_mean(sum_beta_tildes_m, num_sum_beta_m, beta_outlier_mask_m)

            #sum_z_scores_m[beta_outlier_mask_m] = 0
            #avg_z_scores_v = np.sum(sum_z_scores_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_z_scores_v = __outlier_resistant_mean(sum_z_scores_m, num_sum_beta_m, beta_outlier_mask_m)

            self.beta_tildes = avg_beta_tildes_v
            self.z_scores = avg_z_scores_v
            self.p_values = 2*scipy.stats.norm.cdf(-np.abs(self.z_scores))
            self.ses = np.full(self.beta_tildes.shape, 100.0)
            self.ses[self.z_scores != 0] = np.abs(self.beta_tildes[self.z_scores != 0] / self.z_scores[self.z_scores != 0])

            self.betas = avg_betas_v
            self.betas_uncorrected = avg_betas_uncorrected_v
            self.inf_betas = None
            self.non_inf_avg_cond_betas = None
            self.non_inf_avg_postps = avg_postp_v

            #priors_missing is at the end
            self.priors = avg_priors_v
            self.priors_missing = avg_priors_missing_v
            self.combined_Ds_missing = avg_Ds_missing_v

            self.Y_for_regression = avg_bf_orig_v
            self.Y = avg_bf_orig_raw_v

            self.combined_Ds_for_regression = avg_Ds_v
            self.combined_Ds = avg_D_raws_v

            self.combined_prior_Ys_for_regression = avg_log_pos_v - self.background_log_bf
            self.combined_prior_Ys = avg_log_po_raws_v - self.background_log_bf

            #self.combined_prior_Y_ses = avg_log_pos_ses

            gene_N = self.get_gene_N()
            gene_N_missing = self.get_gene_N(get_missing=True)

            all_gene_N = gene_N
            if self.genes_missing is not None:
                assert(gene_N_missing is not None)
                all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

            total_priors = np.concatenate((self.priors, self.priors_missing))
            priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
            priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

            if adjust_priors:
                log("Adjusting priors with slope %.4g" % priors_slope)
                self.priors_adj = self.priors - priors_slope * gene_N - priors_intercept
                if self.genes_missing is not None:
                    self.priors_adj_missing = self.priors_missing - priors_slope * gene_N_missing

                combined_slope = np.cov(self.combined_prior_Ys, gene_N)[0,1] / np.var(gene_N)
                combined_intercept = np.mean(self.combined_prior_Ys - gene_N * combined_slope)

                log("Adjusting combined with slope %.4g" % combined_slope)
                self.combined_prior_Ys_adj = self.combined_prior_Ys - combined_slope * gene_N - combined_intercept

            begin_slice = int(min_num_iter * 0.1)

            if gibbs_good:
                break

    def _append_with_any_user(self, P):
        if P is None:
            return None
        else:
            if sparse.issparse(P):
                P = P.todense()
            return np.hstack((P, 1 - np.prod(1 - P, axis=1)[:,np.newaxis]))

    def _project_H_with_fixed_W(self, W, V_new, P_gene_set, P_gene_new, phi=0.0, lambdak=None, n_iter=100, tol=1e-5, normalize_genes=False, cap_genes=False):
        """
        Projects new genes onto the learned NMF factors W using update rules consistent with the original NMF algorithm.

        Parameters:
        - W: numpy array of shape (N, K), fixed basis matrix from NMF.
        - V_new: numpy array of shape (N, M_new), new genes to project (gene sets x new genes).
        - P_gene_set: numpy array of shape (N, U), gene set weights matrix.
        - P_gene_new: numpy array of shape (M_new, U), gene weights matrix for new genes.
        - phi: regularization parameter (default: 0.0).
        - lambdak: numpy array of shape (K,), ARD weights (default: None). If None, set to ones.
        - n_iter: maximum number of iterations (default: 100).
        - tol: tolerance for convergence (default: 1e-5).

        Returns:
        - H_new: numpy array of shape (M_new, K), the loadings for the new genes.
        """

        eps = 1e-10  # Small constant to prevent division by zero

        N, K = W.shape  # N: number of gene sets, K: number of latent factors
        N_v, M_new = V_new.shape  # N_v should be equal to N
        assert N == N_v, "V_new (%s,%s) must have the same number of rows as W (%s,%s)" % (V_new.shape[0], V_new.shape[1], W.shape[0], W.shape[1])

        if sparse.issparse(V_new):
            V_new = V_new.toarray()

        P_gene_set = self._append_with_any_user(P_gene_set)
        P_gene_new = self._append_with_any_user(P_gene_new)

        use_extended = P_gene_new is not None or P_gene_set is not None
        if use_extended:
            if P_gene_new is not None:
                if P_gene_new.ndim == 1:
                    P_gene_new = P_gene_new[:,np.newaxis]
                U = P_gene_new.shape[1]
            else:
                U = P_gene_set.shape[1] if P_gene_set.ndim > 1 else 1
                P_gene_new = np.ones((M_new, U))

            if P_gene_set is not None:
                if P_gene_set.ndim == 1:
                    P_gene_set = P_gene_set[:,np.newaxis]
                assert P_gene_set.shape[1] == U, "P_gene_new (%s, %s) and P_gene_set (%s, %s) must have the same number of users" % (P_gene_new.shape[0], P_gene_new.shape[1], P_gene_set.shape[0], P_gene_set.shape[1])
            else:
                P_gene_set = np.ones((N, U))

            assert P_gene_set.shape == (N, U), f"P_gene_set should have shape ({N}, {U}), not {P_gene_set.shape}"
            assert P_gene_new.shape == (M_new, U), f"P_gene_new should have shape ({M_new}, {U}), not {P_gene_new.shape}"

            # Compute S_new = P_gene_set @ P_gene_new.T, shape (N, M_new)
            S_new = P_gene_set @ P_gene_new.T  # (N x U) @ (U x M_new) -> (N x M_new)
            if sparse.issparse(S_new):
                S_new = S_new.toarray()

            S_new += eps  # Avoid zeros
        else:
            S_new = np.ones_like(V_new)  # If no weighting, S is a matrix of ones


        # Initialize H_new with positive random values, shape (K, M_new)
        V_max = np.max(V_new)
        H_new = np.random.random((K, M_new)) * V_max

        # Initialize lambdak if not provided
        if lambdak is None:
            lambdak = np.ones(K)
        else:
            lambdak = np.array(lambdak)
            assert lambdak.shape == (K,), "lambdak should have shape (K,)"

        # Initialize V_ap_new
        V_ap_new = W @ H_new  # Shape: (N, M_new)

        for it in range(n_iter):
            # Compute numerator and denominator for H_new

            numerator_H = W.T @ (V_new * S_new)  # Shape: (K, M_new)
            denominator_H = W.T @ (V_ap_new * S_new) + phi * H_new * (1 / lambdak)[:, np.newaxis] + eps  # Shape: (K, M_new)

            # Update H_new
            H_new_update = H_new * (numerator_H / denominator_H)

            # Ensure non-negativity
            H_new_update = np.maximum(H_new_update, 0)

            if normalize_genes:
                H_sum = np.sum(H_new_update, axis=0, keepdims=True)
                H_sum[H_sum < 1] = 1  # Avoid division by zero or small numbers
                H_new_update = H_new_update / H_sum

            if cap_genes:
                H_new_update = np.clip(H_new_update, 0, 1)

            # Check convergence
            diff = np.linalg.norm(H_new_update - H_new, 'fro') / (np.linalg.norm(H_new, 'fro') + eps)
            H_new = H_new_update

            # Update V_ap_new
            V_ap_new = W @ H_new

            if diff < tol:
                break

        return H_new.T


    def _nnls_project_matrix(self, W, X_new, max_iter=500, tol=1e-5, max_value=None, max_sum=None):
        """
        This code was written by GPT-4.

        Parameters:
        - W: numpy array of shape (n_features, n_components), basis matrix from NMF.
        - X_new: numpy array of shape (n_samples, n_features), each row is a new vector to project.
        - max_iter: maximum number of iterations for the multiplicative update.
        - tol: tolerance for convergence.
        - max_value: maximum allowed value for any entry in H_new.

        Returns:
        - H_new: numpy array of shape (n_samples, n_components), the non-negative loadings for each row in X_new.
        """

        orig_vector = False
        if X_new.ndim == 1:
            orig_vector = True
            X_new = X_new[np.newaxis, :]

        # Initialize H_new with random positive values
        n_components = W.shape[1]
        n_samples = X_new.shape[0]
        H_new = np.random.rand(n_samples, n_components)

        # Precompute W^T * W for efficiency
        WT_W = W.T @ W
        if sparse.issparse(WT_W):
            WT_W = WT_W.toarray()

        # Iterative update
        for i in range(max_iter):
            # Compute numerator and denominator
            numerator = X_new @ W
            denominator = H_new @ WT_W + 1e-10  # Small epsilon to avoid division by zero

            # Update H_new
            if sparse.issparse(numerator):
                H_new_update = (numerator.multiply(1.0 / denominator)).multiply(H_new)
                H_new_update = H_new_update.toarray()
            else:
                H_new_update = H_new * (numerator / denominator)

            H_new_update[H_new_update < 0] = 0


            # Apply maximum value cap if specified
            if max_value is not None:
                H_new_update[H_new_update > max_value] = max_value

            # Apply maximum sum cap if specified
            if max_sum is not None:
                H_sums = H_new_update.sum(axis=1)
                above_sum_mask = H_sums > max_sum
                if np.sum(above_sum_mask) > 0:
                    H_new_update[above_sum_mask,:] = (H_new_update[above_sum_mask,:].T / H_sums[above_sum_mask]).T

            # Check for convergence
            norm = np.linalg.norm(H_new_update - H_new, 'fro')

            if norm < tol:
                break

            H_new = H_new_update

        if orig_vector:
            H_new = np.squeeze(H_new, axis=0)

        return H_new

    def _bayes_nmf_l2_extension(self, V0, P_gene_set=None, P_gene=None, n_iter=10000, a0=10, tol=1e-7, K=15,
                                K0=15, phi=1.0, cap_genes=False, normalize_genes=False, cap_gene_sets=False, normalize_gene_sets=False):
        """
        Bayesian NMF with Automatic Relevance Determination (ARD), using Gaussian (L2) likelihood.
        Extended to handle additional weighting matrices P_gene and P_gene_set without materializing large tensors.

        Parameters:
        - V0: Input data matrix (gene sets x genes), containing Poisson rate parameters.
        - P_gene_set: Gene set weights matrix (gene sets x users), shape (N x U), optional.
        - P_gene: Gene weights matrix (genes x users), shape (M x U), optional.
        - Other parameters as before.

        Returns:
        - W: Gene set factor matrix (gene sets x latent factors).
        - H: Gene factor matrix (latent factors x genes).
        - n_like: Final negative log-likelihood.
        - n_evid: Final evidence value.
        - n_lambda: Final ARD weights.
        - n_error: Final reconstruction error.
        """

        eps = 1e-50
        delambda = 1.0

        # Ensure V0 is non-negative
        V = V0 - np.min(V0)
        N, M = V.shape  # Number of gene sets (N) and genes (M)

        # Initialize W and H with positive random values
        Vmax = np.max(V)
        W = np.random.random((N, K)) * Vmax  # Gene sets x latent factors
        H = np.random.random((K, M)) * Vmax  # Latent factors x genes

        V_ap = W @ H + eps  # Initial approximation of V

        # Initialize ARD parameters
        phi = (np.std(V) ** 2) * phi
        C = (N + M) / 2 + a0 + 1
        b0 = 3.14 * (a0 - 1) * np.mean(V) / (2 * K0)
        lambda_bound = b0 / C
        lambdak = (0.5 * (np.sum(W ** 2, axis=0) + np.sum(H ** 2, axis=1)) + b0) / C
        lambda_cut = lambda_bound * 1.5

        n_like = []
        n_evid = []
        n_error = []
        n_lambda = [lambdak]
        it = 1

        P_gene_set = self._append_with_any_user(P_gene_set)
        P_gene = self._append_with_any_user(P_gene)

        # Check if P_gene and P_gene_set are specified
        use_extended = P_gene is not None or P_gene_set is not None
        if use_extended:
            if P_gene is not None:
                if P_gene.ndim == 1:
                    P_gene = P_gene[:,np.newaxis]
                U = P_gene.shape[1]
            else:
                U = P_gene_set.shape[1] if P_gene_set.ndim > 1 else 1
                P_gene = np.ones((M, U))

            if P_gene_set is not None:
                if P_gene_set.ndim == 1:
                    P_gene_set = P_gene_set[:,np.newaxis]
                assert P_gene_set.shape[1] == U, f"P_gene ({P_gene.shape[0]},{P_gene.shape[1]}) and P_gene_set ({P_gene_set.shape[0]},{P_gene_set.shape[1]}) must have the same number of users"
            else:
                P_gene_set = np.ones((N, U))

            assert P_gene_set.shape == (N, U), f"P_gene_set should have shape ({N}, {U}), not {P_gene_set.shape}"
            assert P_gene.shape == (M, U), f"P_gene should have shape ({M}, {U}), not {P_gene.shape}"

            # Compute S = P_gene_set @ P_gene.T (N x U) @ (U x M) -> N x M
            S = P_gene_set @ P_gene.T  # Weighting matrix S of shape (N x M)
        else:
            S = np.ones_like(V)  # If no weighting, S is a matrix of ones

        if sparse.issparse(S):
            S = S.toarray()

        while delambda >= tol and it <= n_iter:
            # Update H
            numerator_H = W.T @ (V * S)
            denominator_H = W.T @ (V_ap * S) + phi * H * (1 / lambdak)[:, np.newaxis] + eps
            H *= numerator_H / denominator_H
            H = np.maximum(H, 0)

            if normalize_genes:
                H_sum = np.sum(H, axis=0, keepdims=True)
                H_sum[H_sum < 1] = 1  # Avoid division by zero or small numbers
                H = H / H_sum
            if cap_genes:
                H = np.clip(H, 0, 1)

            V_ap = W @ H + eps  # Update approximation

            # Update W
            numerator_W = (V * S) @ H.T
            denominator_W = (V_ap * S) @ H.T + phi * W * (1 / lambdak)[np.newaxis, :] + eps
            W *= numerator_W / denominator_W
            W = np.maximum(W, 0)

            if normalize_gene_sets:
                W_sum = np.sum(W, axis=1, keepdims=True)
                W_sum[W_sum < 1] = 1  # Avoid division by zero or small numbers
                W = W / W_sum
            if cap_gene_sets:
                W = np.clip(W, 0, 1)

            V_ap = W @ H + eps  # Update approximation

            # Compute Gaussian negative log-likelihood
            like = np.sum(0.5 * S * (V - V_ap) ** 2)

            # Update ARD weights
            lambdak_new = (0.5 * (np.sum(W ** 2, axis=0) +
                                  np.sum(H ** 2, axis=1)) + b0) / C
            delambda = np.max(np.abs(lambdak_new - lambdak) / (lambdak + eps))
            lambdak = lambdak_new

            # Compute evidence and error
            regularization = phi * np.sum((0.5 * (np.sum(W ** 2, axis=0) +
                                                   np.sum(H ** 2, axis=1)) + b0) / lambdak +
                                          C * np.log(lambdak))
            evid = like + regularization
            error = np.sum(S * (V - V_ap) ** 2)

            n_like.append(like)
            n_evid.append(evid)
            n_lambda.append(lambdak)
            n_error.append(error)

            if it % 100 == 0 or it == 1 or delambda < tol:
                factors = np.sum(np.sum(W, axis=0) != 0)
                factors_non_zero = np.sum(lambdak >= lambda_cut)
                log(f"Iteration={it}; evid={evid:.3g}; lik={like:.3g}; err={error:.3g}; delambda={delambda:.3g}; factors={factors}; factors_non_zero={factors_non_zero}")
            it += 1

        # Return the results
        return W, H, n_like[-1], n_evid[-1], n_lambda[-1], n_error[-1]

    #this code is adapted from https://github.com/gwas-partitioning/bnmf-clustering
    def _bayes_nmf_l2(self, V0, n_iter=10000, a0=10, tol=1e-7, K=15, K0=15, phi=1.0):

        n_iter=100

        # Bayesian NMF with half-normal priors for W and H
        # V0: input z-score matrix (variants x traits)
        # n_iter: Number of iterations for parameter optimization
        # a0: Hyper-parameter for inverse gamma prior on ARD relevance weights
        # tol: Tolerance for convergence of fitting procedure
        # K: Number of clusters to be initialized (algorithm may drive some to zero)
        # K0: Used for setting b0 (lambda prior hyper-parameter) -- should be equal to K
        # phi: Scaling parameter

        eps = 1.e-50
        delambda = 1.0
        #active_nodes = np.sum(V0, axis=0) != 0
        #V0 = V0[:,active_nodes]
        V = V0 - np.min(V0)
        Vmin = np.min(V)
        Vmax = np.max(V)
        N = V.shape[0]
        M = V.shape[1]


        W = np.random.random((N, K)) * Vmax #NxK
        H = np.random.random((K, M)) * Vmax #KxM

        I = np.ones((N, M)) #NxM
        V_ap = W.dot(H) + eps #NxM

        phi = np.power(np.std(V), 2) * phi
        C = (N + M) / 2 + a0 + 1
        b0 = 3.14 * (a0 - 1) * np.mean(V) / (2 * K0)
        lambda_bound = b0 / C
        lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
        lambda_cut = lambda_bound * 1.5


        n_like = [None]
        n_evid = [None]
        n_error = [None]
        n_lambda = [lambdak]
        it = 1
        count = 1
        while delambda >= tol and it < n_iter:

            H = H * (W.T.dot(V)) / (W.T.dot(V_ap) + phi * H * np.repeat(1/lambdak, M).reshape(len(lambdak), M) + eps)
            V_ap = W.dot(H) + eps
            W = W * (V.dot(H.T)) / (V_ap.dot(H.T) + phi * W * np.tile(1/lambdak, N).reshape(N, len(lambdak)) + eps)
            V_ap = W.dot(H) + eps
            lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
            delambda = np.max(np.abs(lambdak - n_lambda[it - 1]) / n_lambda[it - 1])
            like = np.sum(np.power(V - V_ap, 2)) / 2
            n_like.append(like)
            n_evid.append(like + phi * np.sum((0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / lambdak + C * np.log(lambdak)))
            n_lambda.append(lambdak)
            n_error.append(np.sum(np.power(V - V_ap, 2)))
            if it % 100 == 0:
                log("Iteration=%d; evid=%.3g; lik=%.3g; err=%.3g; delambda=%.3g; factors=%d; factors_non_zero=%d" % (it, n_evid[it], n_like[it], n_error[it], delambda, np.sum(np.sum(W, axis=0) != 0), np.sum(lambdak >= lambda_cut)), TRACE)
            it += 1

        W[W < eps] = 0
        H[H < eps] = 0

        return W, H, n_like[-1], n_evid[-1], n_lambda[-1], n_error[-1]
            #W # Variant weight matrix (N x K)
            #H # Trait weight matrix (K x M)
            #n_like # List of reconstruction errors (sum of squared errors / 2) per iteration
            #n_evid # List of negative log-likelihoods per iteration
            #n_lambda # List of lambda vectors (shared weights for each of K clusters, some ~0) per iteration
            #n_error # List of reconstruction errors (sum of squared errors) per iteration


    def num_factors(self):
        if self.exp_lambdak is None:
            return 0
        else:
            return len(self.exp_lambdak)

    def run_factor(self, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, keep_original_loadings=False):

        if self.X_orig is None:
            bail("Cannot run factoring without X")

        if (anchor_any_gene or anchor_any_pheno or anchor_gene_set or anchor_gene_mask is not None or anchor_pheno_mask is not None or pheno_prune_value is not None or pheno_prune_number is not None) and self.X_phewas_beta is None:
            bail("Cannot run factoring without X phewas")

        if anchor_any_gene:
            if anchor_any_pheno:
                warn("Ignoring anchor any pheno since anchor any gene was specified")
            if anchor_gene_mask:
                warn("Ignoring anchor gene since anchor any gene was specified")
            if anchor_pheno_mask:
                warn("Ignoring anchor pheno since anchor any gene was specified")
            if anchor_gene_set:
                warn("Ignoring anchor gene set since anchor any gene was specified")

            self._record_params({"anchor": "any_gene"})
            anchor_any_pheno = False
            anchor_pheno_mask = None
            anchor_gene_mask = np.full(self.X_orig.shape[0], True)
            anchor_gene_set = False

        elif anchor_any_pheno:
            if anchor_gene_mask:
                warn("Ignoring anchor gene since anchor any pheno was specified")
            if anchor_pheno_mask:
                warn("Ignoring anchor pheno since anchor any pheno was specified")
            if anchor_gene_set:
                warn("Ignoring anchor gene set since anchor any pheno was specified")
            anchor_gene_mask = None
            anchor_pheno_mask = np.full(self.X_phewas_beta.shape[0], True)
            anchor_gene_set = False
            self._record_params({"anchor": "any_pheno"})
        elif anchor_gene_set:
            if anchor_gene_mask:
                warn("Ignoring anchor gene since anchor gene set was specified")
            if anchor_pheno_mask:
                warn("Ignoring anchor pheno since anchor gene set was specified")
            anchor_gene_mask = None
            anchor_pheno_mask = None
            self._record_params({"anchor": "gene set"})

        #ensure at most one anchor mask, and initialize the matrix mask accordingly
        #remember that single pheno anchoring mode is implicit and doesn't have the anchor mask defined
        num_users = 1
        anchor_mask = None
        factor_gene_set_x_pheno = False
        pheno_Y = None

        if anchor_gene_mask is not None or anchor_gene_set:
            if anchor_pheno_mask is not None:
                warn("Ignoring anchor pheno since anchor gene or anchor gene set was specified")
                anchor_pheno_mask = None
            gene_or_pheno_mask = np.full(self.X_phewas_beta.shape[0], True)
            gene_set_mask = np.full(self.X_phewas_beta.shape[1], True)
            factor_gene_set_x_pheno = True

            combined_prior_Ys = self.gene_pheno_combined_prior_Ys.T if self.gene_pheno_combined_prior_Ys is not None else None
            priors = self.gene_pheno_priors.T if self.gene_pheno_priors is not None else None
            Y = self.gene_pheno_Y.T if self.gene_pheno_Y is not None else None

            self._record_params({"factor_gene_vectors": "gene_pheno.T"})

            if anchor_gene_mask is not None:
                betas = None
                betas_uncorrected = None

                anchor_mask = anchor_gene_mask
                num_users = np.sum(anchor_mask)
                self._record_params({"factor_gene_set_vectors": "None"})

            else:
                #we need to set things up below
                #we are going to construct a pheno x gene set matrix, using the X_phewas as input
                #we need to have weights for the rows (phenos) and columns (gene sets)
                #the column weights need to be the betas

                anchor_gene_mask = np.full(1, True)
                anchor_mask = anchor_gene_mask
                num_users = 1

                #for the gene set mode, we use the pheno_Y for weights, and do a special setting below
                #we need to keep combined_prior_Y for projecting, but use pheno_Y for weighting
                pheno_Y = self.pheno_Y_vs_input_combined_prior_Ys_beta if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None else self.pheno_Y_vs_input_Y_beta if self.pheno_Y_vs_input_Y_beta is not None else self.pheno_Y_vs_input_priors_beta
                if pheno_Y is not None:
                    pheno_Y = pheno_Y[:,np.newaxis]
                
                #betas are in external units
                betas = (self.betas / self.scale_factors)[:,np.newaxis] if self.betas is not None else None
                betas_uncorrected = (self.betas_uncorrected / self.scale_factors)[:,np.newaxis] if self.betas_uncorrected is not None else None
                self._record_params({"factor_gene_set_vectors": "betas"})

        else:
            if anchor_pheno_mask is not None and anchor_gene_mask is not None:
                warn("Ignoring anchor gene since anchor pheno was specified")
            anchor_gene_mask = None
            gene_or_pheno_mask = np.full(self.X_orig.shape[0], True)
            gene_set_mask = np.full(self.X_orig.shape[1], True)
            if anchor_pheno_mask is not None:

                anchor_mask = anchor_pheno_mask

                combined_prior_Ys = self.gene_pheno_combined_prior_Ys
                priors = self.gene_pheno_priors
                Y = self.gene_pheno_Y

                self._record_params({"factor_gene_vectors": "gene_pheno"})
                betas = self.X_phewas_beta.T if self.X_phewas_beta is not None else None
                betas_uncorrected = self.X_phewas_beta_uncorrected.T if self.X_phewas_beta_uncorrected is not None else None
                self._record_params({"factor_gene_set_vectors": "X_phewas"})

            else:

                combined_prior_Ys = self.combined_prior_Ys[:,np.newaxis] if self.combined_prior_Ys is not None else None
                priors = self.priors[:,np.newaxis] if self.priors is not None else None
                Y = self.Y[:,np.newaxis] if self.Y is not None else None

                self._record_params({"factor_gene_vectors": "Y"})

                betas = (self.betas / self.scale_factors)[:,np.newaxis] if self.betas is not None else None
                betas_uncorrected = (self.betas_uncorrected / self.scale_factors)[:,np.newaxis] if self.betas_uncorrected is not None else None

                self._record_params({"factor_gene_set_vectors": "betas"})


                #when running the original factoring based off the internal betas and gene scores, we are going to emulate the phewas-like behavior by appending these as the only anchor alongside any gene/pheno loaded values
                #this will allow projection to other phenotypes to happen naturally below
                anchor_mask = np.full(1, True)

                have_phewas = False
                if combined_prior_Ys is not None and self.gene_pheno_combined_prior_Ys is not None:
                    combined_prior_Ys = sparse.hstack((self.gene_pheno_combined_prior_Ys, sparse.csc_matrix(combined_prior_Ys))).tocsc()
                    have_phewas = True
                if priors is not None and self.gene_pheno_priors is not None:
                    priors = sparse.hstack((self.gene_pheno_priors, sparse.csc_matrix(priors))).tocsc()
                    have_phewas = True
                if Y is not None and self.gene_pheno_Y is not None:
                    Y = sparse.hstack((self.gene_pheno_Y, sparse.csc_matrix(Y))).tocsc()
                    have_phewas = True

                if betas is not None and self.X_phewas_beta is not None:
                    betas = sparse.hstack((self.X_phewas_beta.T, sparse.csc_matrix(betas))).tocsc()
                    have_phewas = True
                if betas_uncorrected is not None and self.X_phewas_beta_uncorrected is not None:
                    betas_uncorrected = sparse.hstack((self.X_phewas_beta_uncorrected.T, sparse.csc_matrix(betas_uncorrected))).tocsc()
                    have_phewas = True

                if have_phewas:
                    #we have phewas for at least one of combined, prior, or Y
                    #set those that don't to None
                    #otherwise update the internal structures
                    if combined_prior_Ys is not None and combined_prior_Ys.shape[1] == 1:
                        combined_prior_Ys = None
                    else:
                        self.gene_pheno_combined_prior_Ys = combined_prior_Ys
                        
                    if priors is not None and priors.shape[1] == 1:
                        priors = None
                    else:
                        self.gene_pheno_priors = priors

                    if Y is not None and Y.shape[1] == 1:
                        Y = None
                    else:
                        self.gene_pheno_Y = Y
                    if betas is not None and betas.shape[1] == 1:
                        betas = None
                    else:
                        self.X_phewas_beta = betas.T
                    if betas_uncorrected is not None and betas_uncorrected.shape[1] == 1:
                        betas_uncorrected = None
                    else:
                        self.X_phewas_beta_uncorrected = betas_uncorrected.T

                    self.phenos.append("__default__")
                    self.default_pheno_mask = np.append(np.full(len(self.phenos), False), True)

                    #we need to update these as well
                    self.pheno_Y_vs_input_Y_beta = np.append(self.pheno_Y_vs_input_Y_beta, 0) if self.pheno_Y_vs_input_Y_beta is not None else None
                    self.pheno_Y_vs_input_Y_beta_tilde = np.append(self.pheno_Y_vs_input_Y_beta_tilde, 0) if self.pheno_Y_vs_input_Y_beta_tilde is not None else None
                    self.pheno_Y_vs_input_Y_se = np.append(self.pheno_Y_vs_input_Y_se, 0) if self.pheno_Y_vs_input_Y_se is not None else None
                    self.pheno_Y_vs_input_Y_Z = np.append(self.pheno_Y_vs_input_Y_Z, 0) if self.pheno_Y_vs_input_Y_Z is not None else None
                    self.pheno_Y_vs_input_Y_p_value = np.append(self.pheno_Y_vs_input_Y_p_value, 1) if self.pheno_Y_vs_input_Y_p_value is not None else None

                    self.pheno_combined_prior_Ys_vs_input_Y_beta = np.append(self.pheno_combined_prior_Ys_vs_input_Y_beta, 0) if self.pheno_combined_prior_Ys_vs_input_Y_beta is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde = np.append(self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde, 0) if self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_se = np.append(self.pheno_combined_prior_Ys_vs_input_Y_se, 0) if self.pheno_combined_prior_Ys_vs_input_Y_se is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_Z = np.append(self.pheno_combined_prior_Ys_vs_input_Y_Z, 0) if self.pheno_combined_prior_Ys_vs_input_Y_Z is not None else None
                    self.pheno_combined_prior_Ys_vs_input_Y_p_value = np.append(self.pheno_combined_prior_Ys_vs_input_Y_p_value, 1) if self.pheno_combined_prior_Ys_vs_input_Y_p_value is not None else None

                    self.pheno_Y_vs_input_combined_prior_Ys_beta = np.append(self.pheno_Y_vs_input_combined_prior_Ys_beta, 0) if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde = np.append(self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde, 0) if self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_se = np.append(self.pheno_Y_vs_input_combined_prior_Ys_se, 0) if self.pheno_Y_vs_input_combined_prior_Ys_se is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_Z = np.append(self.pheno_Y_vs_input_combined_prior_Ys_Z, 0) if self.pheno_Y_vs_input_combined_prior_Ys_Z is not None else None
                    self.pheno_Y_vs_input_combined_prior_Ys_p_value = np.append(self.pheno_Y_vs_input_combined_prior_Ys_p_value, 1) if self.pheno_Y_vs_input_combined_prior_Ys_p_value is not None else None

                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z, 0) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z is not None else None
                    self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value = np.append(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value, 1) if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value is not None else None

                    self.pheno_Y_vs_input_priors_beta = np.append(self.pheno_Y_vs_input_priors_beta, 0) if self.pheno_Y_vs_input_priors_beta is not None else None
                    self.pheno_Y_vs_input_priors_beta_tilde = np.append(self.pheno_Y_vs_input_priors_beta_tilde, 0) if self.pheno_Y_vs_input_priors_beta_tilde is not None else None
                    self.pheno_Y_vs_input_priors_se = np.append(self.pheno_Y_vs_input_priors_se, 0) if self.pheno_Y_vs_input_priors_se is not None else None
                    self.pheno_Y_vs_input_priors_Z = np.append(self.pheno_Y_vs_input_priors_Z, 0) if self.pheno_Y_vs_input_priors_Z is not None else None
                    self.pheno_Y_vs_input_priors_p_value = np.append(self.pheno_Y_vs_input_priors_p_value, 1) if self.pheno_Y_vs_input_priors_p_value is not None else None

                    self.pheno_combined_prior_Ys_vs_input_priors_beta = np.append(self.pheno_combined_prior_Ys_vs_input_priors_beta, 0) if self.pheno_combined_prior_Ys_vs_input_priors_beta is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde = np.append(self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde, 0) if self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_se = np.append(self.pheno_combined_prior_Ys_vs_input_priors_se, 0) if self.pheno_combined_prior_Ys_vs_input_priors_se is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_Z = np.append(self.pheno_combined_prior_Ys_vs_input_priors_Z, 0) if self.pheno_combined_prior_Ys_vs_input_priors_Z is not None else None
                    self.pheno_combined_prior_Ys_vs_input_priors_p_value = np.append(self.pheno_combined_prior_Ys_vs_input_priors_p_value, 1) if self.pheno_combined_prior_Ys_vs_input_priors_p_value is not None else None

                    if combined_prior_Ys is None and priors is None and Y is None:
                        bail("Need to load gene phewas stats if you are loading gene set phewas stats")
                    if betas is None and betas_uncorrected is None:
                        bail("Need to load gene set phewas stats if you are loading gene phewas stats")
                    
                #the newly appended ones are not anchors
                anchor_mask = np.append(np.full((combined_prior_Ys.shape[1] if combined_prior_Ys is not None else priors.shape[1] if priors is not None else Y.shape[1] if Y is not None else 1) - 1, False), anchor_mask)


            num_users = np.sum(anchor_pheno_mask)

        #get one dimensional vectors with probabilities
        gene_or_pheno_full_vector = combined_prior_Ys if combined_prior_Ys is not None else priors if priors is not None else Y if Y is not None else None

        gene_or_pheno_vector = None
        if anchor_gene_set:
            gene_or_pheno_vector = pheno_Y
        else:
            if gene_or_pheno_full_vector is not None:
                gene_or_pheno_vector = gene_or_pheno_full_vector[:,anchor_mask]

        if gene_or_pheno_vector is not None:
            if sparse.issparse(gene_or_pheno_vector):
                gene_or_pheno_vector = gene_or_pheno_vector.toarray()

        gene_or_pheno_filter_type = "combined_prior_Ys" if combined_prior_Ys is not None else "priors" if priors is not None else "Y" if Y is not None else None        

        #now get the aggregations and masks
        gene_or_pheno_max_vector = np.max(gene_or_pheno_vector, axis=1) if gene_or_pheno_vector is not None else None

        if gene_or_pheno_max_vector is not None and gene_or_pheno_filter_value is not None:
            gene_or_pheno_mask = gene_or_pheno_max_vector > gene_or_pheno_filter_value

        def __combine_prune_masks(prune_masks, prune_number, sort_rank, tag):
            if prune_masks is None or len(prune_masks) == 0:
                return None
            all_prune_mask = np.full(len(prune_masks[0]), False)
            for cur_prune_mask in prune_masks:
                all_prune_mask[cur_prune_mask] = True
                log("Adding %d relatively uncorrelated %ss (total now %d)" % (np.sum(cur_prune_mask), tag, np.sum(all_prune_mask)), TRACE)
                if np.sum(all_prune_mask) > prune_number:
                    break
            if np.sum(all_prune_mask) > prune_number:
                threshold_value = sorted(sort_rank[all_prune_mask])[prune_number - 1]
                all_prune_mask[sort_rank > threshold_value] = False
            if np.sum(~all_prune_mask) > 0:
                log("Found %d %ss remaining after pruning to max number (of %d)" % (np.sum(all_prune_mask), tag, len(self.phenos)))
            return all_prune_mask

        if pheno_prune_value is not None or pheno_prune_number is not None:
            mask_for_pruning = gene_or_pheno_mask if factor_gene_set_x_pheno else anchor_pheno_mask
            if mask_for_pruning is not None:
            
                if factor_gene_set_x_pheno:
                    log("Pruning phenos to reduce matrix size", DEBUG)
                else:
                    log("Pruning phenos to reduce number of anchors", DEBUG)                    

                pheno_sort_rank = -self.X_phewas_beta.mean(axis=1).A1 if self.X_phewas_beta is not None else np.arange(len(mask_for_pruning))
                #now if we request pruning
                if pheno_prune_value is not None:
                    pheno_prune_mask = self._prune_gene_sets(pheno_prune_value, X_orig=self.X_phewas_beta_uncorrected[mask_for_pruning,:].T, gene_sets=[self.phenos[i] for i in np.where(mask_for_pruning)[0]], rank_vector=pheno_sort_rank[mask_for_pruning], do_internal_pruning=False)
                    log("Found %d phenos remaining after pruning (of %d)" % (np.sum(pheno_prune_mask), len(self.phenos)))

                    mask_for_pruning[np.where(mask_for_pruning)[0][~pheno_prune_mask]] = False

                if pheno_prune_number is not None:
                    (mean_shifts, scale_factors) = self._calc_X_shift_scale(self.X_phewas_beta_uncorrected[mask_for_pruning,:].T)
                    pheno_prune_number_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_phewas_beta_uncorrected[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=pheno_sort_rank[mask_for_pruning], stop_at=pheno_prune_number, tag="phenos")
                    all_pheno_prune_mask = __combine_prune_masks(pheno_prune_number_masks, pheno_prune_number, pheno_sort_rank[mask_for_pruning], "pheno")
                    mask_for_pruning[np.where(mask_for_pruning)[0][~all_pheno_prune_mask]] = False
                if mask_for_pruning is anchor_pheno_mask and num_users > 1:
                    #in this case, we may have changed the number of users
                    num_users = np.sum(anchor_pheno_mask)

        if not anchor_gene_set and (gene_prune_value is not None or gene_prune_number is not None):
            mask_for_pruning = gene_or_pheno_mask if not factor_gene_set_x_pheno else anchor_gene_mask
            if mask_for_pruning is not None:
                gene_sort_rank = -self.combined_prior_Ys if self.combined_prior_Ys is not None else -self.Y if self.Y is not None else -self.priors if self.priors is not None else np.arange(len(mask_for_pruning))
                if not factor_gene_set_x_pheno:
                    log("Pruning genes to reduce matrix size", DEBUG)
                else:
                    log("Pruning genes to reduce number of anchors", DEBUG)                    


                #now if we request pruning
                if gene_prune_value is not None:
                    gene_prune_mask = self._prune_gene_sets(gene_prune_value, X_orig=self.X_orig[mask_for_pruning,:].T, gene_sets=[self.genes[i] for i in np.where(mask_for_pruning)[0]], rank_vector=gene_sort_rank[mask_for_pruning], do_internal_pruning=False)
                    log("Found %d genes remaining after pruning (of %d)" % (np.sum(gene_prune_mask), len(self.genes)))

                    mask_for_pruning[np.where(mask_for_pruning)[0][~gene_prune_mask]] = False

                if gene_prune_number is not None:
                    (mean_shifts, scale_factors) = self._calc_X_shift_scale(self.X_orig[mask_for_pruning,:].T)
                    gene_prune_number_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=gene_sort_rank[mask_for_pruning], stop_at=gene_prune_number, tag="genes")
                    all_gene_prune_mask = __combine_prune_masks(gene_prune_number_masks, gene_prune_number, gene_sort_rank[mask_for_pruning], "gene")
                    mask_for_pruning[np.where(mask_for_pruning)[0][~all_gene_prune_mask]] = False

                if mask_for_pruning is anchor_gene_mask and num_users > 1:
                    #in this case, we may have changed the number of users
                    num_users = np.sum(anchor_gene_mask)

        #add in the any vectors
        gene_or_pheno_full_prob_vector = None
        if gene_or_pheno_full_vector is not None:
            #we are going to approximate things below the threshold as zero probability, and not fold those in the background prior
            #to get around this we would have to use a dense matrix
            if sparse.issparse(gene_or_pheno_full_vector):
                gene_or_pheno_full_prob_vector_data = np.exp(gene_or_pheno_full_vector.data + self.background_log_bf)
                gene_or_pheno_full_prob_vector_data = gene_or_pheno_full_prob_vector_data / (1 + gene_or_pheno_full_prob_vector_data)
                gene_or_pheno_full_prob_vector = copy.copy(gene_or_pheno_full_vector)
                gene_or_pheno_full_prob_vector.data = gene_or_pheno_full_prob_vector_data
            else:
                gene_or_pheno_full_prob_vector = np.exp(gene_or_pheno_full_vector + self.background_log_bf) / (1 + np.exp(gene_or_pheno_full_vector + self.background_log_bf))

        if anchor_gene_set:
            gene_or_pheno_prob_vector = np.exp(gene_or_pheno_vector + self.background_log_bf) / (1 + np.exp(gene_or_pheno_vector + self.background_log_bf)) if gene_or_pheno_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))
        else:
            gene_or_pheno_prob_vector = gene_or_pheno_full_prob_vector[:,anchor_mask] if gene_or_pheno_full_prob_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))

        if gene_or_pheno_prob_vector is not None and sparse.issparse(gene_or_pheno_prob_vector):
            gene_or_pheno_prob_vector = gene_or_pheno_prob_vector.toarray()

        if anchor_any_gene or anchor_any_pheno:
            #only have one user
            gene_or_pheno_any_prob_vector = 1 - np.prod(1 - gene_or_pheno_prob_vector, axis=1)
            gene_or_pheno_prob_vector = gene_or_pheno_any_prob_vector[:,np.newaxis]

        if factor_gene_set_x_pheno:
            self.pheno_prob_factor_vector = gene_or_pheno_prob_vector
            self.gene_prob_factor_vector = None
        else:
            self.gene_prob_factor_vector = gene_or_pheno_prob_vector
            self.pheno_prob_factor_vector = None

        #now do the gene set vectors and masks
        #normalize
        gene_set_full_vector = betas_uncorrected if betas_uncorrected is not None else betas
        gene_set_vector = None
        if gene_set_full_vector is not None:
            gene_set_vector = gene_set_full_vector[:,anchor_mask]
            if sparse.issparse(gene_set_vector):
                gene_set_vector = gene_set_vector.toarray()

        gene_set_filter_type = "betas_uncorrected" if betas_uncorrected is not None else "betas"
        gene_set_max_vector = np.max(gene_set_vector, axis=1) if gene_set_vector is not None else None

        if gene_set_max_vector is not None and gene_set_filter_value is not None:
            gene_set_mask = gene_set_max_vector > gene_set_filter_value


        gene_set_sort_rank = -(self.X_phewas_beta_uncorrected.mean(axis=0).A1 if self.X_phewas_beta_uncorrected is not None else self.betas)

        if gene_set_prune_value is not None or gene_set_prune_number is not None:
            log("Pruning gene sets to reduce matrix size", DEBUG)

        if gene_set_prune_value is not None:
            gene_set_prune_mask = self._prune_gene_sets(gene_set_prune_value, X_orig=self.X_orig[:,gene_set_mask], gene_sets=[self.gene_sets[i] for i in np.where(gene_set_mask)[0]], rank_vector=gene_set_sort_rank[gene_set_mask], do_internal_pruning=False)
            log("Found %d gene_sets remaining after pruning (of %d)" % (np.sum(gene_set_prune_mask), len(self.gene_sets)))
            gene_set_mask[np.where(gene_set_mask)[0][~gene_set_prune_mask]] = False

        if gene_set_prune_number is not None:
            gene_set_prune_number_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,gene_set_mask], mean_shifts=self.mean_shifts[gene_set_mask], scale_factors=self.scale_factors[gene_set_mask], sort_values=gene_set_sort_rank[gene_set_mask], stop_at=pheno_prune_number, tag="gene sets")

            all_gene_set_prune_mask = __combine_prune_masks(gene_set_prune_number_masks, gene_set_prune_number, gene_set_sort_rank[gene_set_mask], "gene set")

            gene_set_mask[np.where(gene_set_mask)[0][~all_gene_set_prune_mask]] = False
        
        gene_set_full_prob_vector = None
        if gene_set_full_vector is not None:
            if sparse.issparse(gene_set_full_vector):
                gene_set_full_prob_vector_data = np.exp(gene_set_full_vector.data + self.background_log_bf)
                gene_set_full_prob_vector_data = gene_set_full_prob_vector_data / (1 + gene_set_full_prob_vector_data)
                gene_set_full_prob_vector = copy.copy(gene_set_full_vector)
                gene_set_full_prob_vector.data = gene_set_full_prob_vector_data
            else:
                gene_set_full_prob_vector = np.exp(gene_set_full_vector + self.background_log_bf) / (1 + np.exp(gene_set_full_vector + self.background_log_bf))

        gene_set_prob_vector = gene_set_full_prob_vector[:,anchor_mask] if gene_set_full_prob_vector is not None else np.ones((len(gene_set_mask), num_users))

        if gene_set_prob_vector is not None and sparse.issparse(gene_set_prob_vector):
            gene_set_prob_vector = gene_set_prob_vector.toarray()

        if anchor_any_gene or anchor_any_pheno:
            #only have one user
            gene_set_any_prob_vector = 1 - np.prod(1 - gene_set_prob_vector, axis=1)
            gene_set_prob_vector = gene_set_any_prob_vector[:,np.newaxis]

        self.gene_set_prob_vector = gene_set_full_prob_vector

        self._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "phi": phi, "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_or_pheno_filter_type": gene_or_pheno_filter_type, "gene_or_pheno_filter_value": gene_or_pheno_filter_value, "pheno_prune_value": pheno_prune_value, "pheno_prune_number": pheno_prune_number, "gene_set_prune_value": gene_set_prune_value, "gene_set_prune_number": gene_set_prune_number, "run_transpose": run_transpose})


        matrix = self.X_phewas_beta_uncorrected.T if factor_gene_set_x_pheno else self.X_orig.T

        matrix = matrix[gene_set_mask,:][:,gene_or_pheno_mask]
        matrix[matrix < 0] = 0

        if not run_transpose:
            matrix = matrix.T

        log("Running matrix factorization")
        if np.sum(~gene_or_pheno_mask) > 0 or np.sum(~gene_set_mask) > 0:
            log("Filtered original matrix from (%s, %s) to (%s, %s)" % (len(gene_or_pheno_mask), len(gene_set_mask), sum(gene_or_pheno_mask), sum(gene_set_mask)))
        log("Matrix to factor shape: (%s, %s)" % (matrix.shape), DEBUG)

        if np.max(matrix.shape) == 0:
            log("Skipping factoring since there aren't enough significant genes and gene sets")
            return

        if np.min(matrix.shape) == 0:
            log("Empty genes or gene sets! Skipping factoring")
            return

        #constrain loadings to be at most 1, but don't require them to sum to 1
        normalize_genes = False
        normalize_gene_sets = False
        cap = True

        result = self._bayes_nmf_l2_extension(matrix.toarray(), gene_set_prob_vector[gene_set_mask,:], gene_or_pheno_prob_vector[gene_or_pheno_mask,:], a0=alpha0, K=max_num_factors, tol=rel_tol, phi=phi, cap_genes=cap, cap_gene_sets=cap, normalize_genes=normalize_genes, normalize_gene_sets=normalize_gene_sets)

        self.exp_lambdak = result[4]
        exp_gene_or_pheno_factors = result[1].T
        self.exp_gene_set_factors = result[0]

        #subset_out the weak factors
        factor_mask = (self.exp_lambdak > min_lambda_threshold) & (np.sum(exp_gene_or_pheno_factors, axis=0) > min_lambda_threshold) & (np.sum(self.exp_gene_set_factors, axis=0) > min_lambda_threshold)
        factor_mask = factor_mask & (np.max(self.exp_gene_set_factors, axis=0) > 1e-5 * np.max(self.exp_gene_set_factors))
        if np.sum(~factor_mask) > 0:
            self.exp_lambdak = self.exp_lambdak[factor_mask]
            exp_gene_or_pheno_factors = exp_gene_or_pheno_factors[:,factor_mask]
            self.exp_gene_set_factors = self.exp_gene_set_factors[:,factor_mask]

        if factor_gene_set_x_pheno:
            self.pheno_factor_pheno_mask = gene_or_pheno_mask
            self.exp_pheno_factors = exp_gene_or_pheno_factors
            self.pheno_prob_factor_vector = gene_or_pheno_prob_vector
            self.gene_prob_factor_vector = None
        else:
            self.gene_factor_gene_mask = gene_or_pheno_mask            
            self.exp_gene_factors = exp_gene_or_pheno_factors
            self.gene_prob_factor_vector = gene_or_pheno_prob_vector
            self.pheno_prob_factor_vector = None

        self.gene_set_prob_factor_vector = gene_set_prob_vector
        self.gene_set_factor_gene_set_mask = gene_set_mask

        #now project the additional genes/phenos/gene sets onto the factors

        log("Projecting factors", TRACE)

        #first get the probabilities for either the genotypes or phenotypes (whichever we didn't use to factor)
        #these need to be specific to the anchors
        if factor_gene_set_x_pheno:
            if gene_or_pheno_full_prob_vector is not None:
                self.gene_prob_factor_vector = self._nnls_project_matrix(self.pheno_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
                self._record_params({"factor_gene_prob_from": "phenos"})
            else:
                self.gene_prob_factor_vector = self._nnls_project_matrix(self.gene_set_prob_factor_vector, self.X_orig)
                self._record_params({"factor_gene_prob_from": "gene_sets"})
        else:
            if gene_or_pheno_full_prob_vector is not None:
                self.pheno_prob_factor_vector = self._nnls_project_matrix(self.gene_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
                self._record_params({"factor_pheno_prob_from": "genes"})
            elif self.X_phewas_beta_uncorrected is not None:
                self.pheno_prob_factor_vector = self._nnls_project_matrix(self.gene_set_prob_factor_vector, self.X_phewas_beta_uncorrected)
                self._record_params({"factor_pheno_prob_from": "gene_sets"})

        if self.gene_set_prob_factor_vector is not None and sparse.issparse(self.gene_set_prob_factor_vector):
            self.gene_set_prob_factor_vector = self.gene_set_prob_factor_vector.toarray()
        if self.gene_prob_factor_vector is not None and sparse.issparse(self.gene_prob_factor_vector):
            self.gene_prob_factor_vector = self.gene_prob_factor_vector.toarray()
        if self.pheno_prob_factor_vector is not None and sparse.issparse(self.pheno_prob_factor_vector):
            self.pheno_prob_factor_vector = self.pheno_prob_factor_vector.toarray()

        gene_matrix_to_project = self.X_orig.T
        if not run_transpose:
            gene_matrix_to_project = gene_matrix_to_project.T

        #all gene factor values
        full_gene_factor_values = self._project_H_with_fixed_W(self.exp_gene_set_factors, gene_matrix_to_project[self.gene_set_factor_gene_set_mask,:], self.gene_set_prob_factor_vector[self.gene_set_factor_gene_set_mask,:], self.gene_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
        if not factor_gene_set_x_pheno and keep_original_loadings:
            full_gene_factor_values[self.gene_factor_gene_mask,:] = self.exp_gene_factors

        #all pheno factor values, either from the phewas used to factor or the phewas passed in to project
        full_pheno_factor_values = self.exp_pheno_factors
        pheno_matrix_to_project = None
        if self.X_phewas_beta_uncorrected is not None and self.pheno_prob_factor_vector is not None:
            pheno_matrix_to_project = self.X_phewas_beta_uncorrected.T
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T

            full_pheno_factor_values = self._project_H_with_fixed_W(self.exp_gene_set_factors, pheno_matrix_to_project[self.gene_set_factor_gene_set_mask,:], self.gene_set_prob_factor_vector[self.gene_set_factor_gene_set_mask,:], self.pheno_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
            if keep_original_loadings:
                full_pheno_factor_values[self.pheno_factor_pheno_mask,:] = self.exp_pheno_factors

        #now gene set factor values, projecting from either phenos or genes depending on what was used
        if factor_gene_set_x_pheno and pheno_matrix_to_project is not None:
            #we have to swap the gene sets and genes, which means transposing the matrix to project and swapping the prios
            full_gene_set_factor_values = self._project_H_with_fixed_W(self.exp_pheno_factors, pheno_matrix_to_project[:,self.pheno_factor_pheno_mask].T if run_transpose else pheno_matrix_to_project[self.pheno_factor_pheno_mask,:].T, self.pheno_prob_factor_vector[self.pheno_factor_pheno_mask,:], self.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)
        else:
            full_gene_set_factor_values = self._project_H_with_fixed_W(self.exp_gene_factors, gene_matrix_to_project[:,self.gene_factor_gene_mask].T if run_transpose else gene_matrix_to_project[self.gene_factor_gene_mask,:].T, self.gene_prob_factor_vector[self.gene_factor_gene_mask,:], self.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)

        if keep_original_loadings:
            full_gene_set_factor_values[self.gene_set_factor_gene_set_mask,:] = self.exp_gene_set_factors

        #update these to store the imputed as well
        self.exp_gene_factors = full_gene_factor_values
        self.exp_pheno_factors = full_pheno_factor_values
        self.exp_gene_set_factors = full_gene_set_factor_values

        if factor_gene_set_x_pheno:
            exp_gene_or_pheno_factors = self.exp_pheno_factors
        else:
            exp_gene_or_pheno_factors = self.exp_gene_factors

        #now update relevance

        matrix_to_mult = self.exp_pheno_factors if factor_gene_set_x_pheno else self.exp_gene_factors
        vector_to_mult = self.pheno_prob_factor_vector if factor_gene_set_x_pheno else self.gene_prob_factor_vector

        #matrix_to_mult: (genes, factors)
        #vector_to_mult: (users, genes)
        #want: (factors, users)

        self.factor_anchor_relevance = self._nnls_project_matrix(matrix_to_mult, vector_to_mult.T, max_value=1).T
        self.factor_relevance = self._nnls_project_matrix(matrix_to_mult, 1 - np.prod(1 - vector_to_mult, axis=1).T, max_value=1).T

        #gene scores are either for phenos or for genes depending on the mode
        reorder_inds = np.argsort(-self.factor_relevance)

        self.exp_lambdak = self.exp_lambdak[reorder_inds]
        self.factor_anchor_relevance = self.factor_anchor_relevance[reorder_inds,:]
        self.factor_relevance = self.factor_relevance[reorder_inds]
        if self.exp_gene_factors is not None:
            self.exp_gene_factors = self.exp_gene_factors[:,reorder_inds]
        if self.exp_pheno_factors is not None:
            self.exp_pheno_factors = self.exp_pheno_factors[:,reorder_inds]
        self.exp_gene_set_factors = self.exp_gene_set_factors[:,reorder_inds]

        #zero out very low values
        threshold = 1e-5
        if self.num_factors() > 0:
            self.exp_gene_factors[self.exp_gene_factors < np.max(self.exp_gene_factors) * threshold] = 0
            if self.exp_pheno_factors is not None:
                self.exp_pheno_factors[self.exp_pheno_factors < np.max(self.exp_pheno_factors) * threshold] = 0
            self.exp_gene_set_factors[self.exp_gene_set_factors < np.max(self.exp_gene_set_factors) * threshold] = 0

        num_top = 5

        #matries are gene x factor
        #materialize matrix of factor x gene x user, then take argmax over axis 1, then swap axes to get gene x factor x user
        

        #(all_genes, factors)
        #(anchor_genes, users)

        top_anchor_gene_or_pheno_inds = np.swapaxes(np.argsort(-(self.exp_pheno_factors if factor_gene_set_x_pheno else self.exp_gene_factors).T[:,:,np.newaxis] * (self.pheno_prob_factor_vector if factor_gene_set_x_pheno else self.gene_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

        top_anchor_gene_set_inds = np.swapaxes(np.argsort(-self.exp_gene_set_factors.T[:,:,np.newaxis] * self.gene_set_prob_factor_vector[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

        #sort by maximum across phenos
        sort_max_across_phenos = True
        if sort_max_across_phenos:

            top_gene_or_pheno_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((self.exp_pheno_factors if factor_gene_set_x_pheno else self.exp_gene_factors).T[:,:,np.newaxis] * (self.pheno_prob_factor_vector if factor_gene_set_x_pheno else self.gene_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)

            top_gene_set_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - (self.exp_gene_set_factors.T[:,:,np.newaxis] * self.gene_set_prob_factor_vector[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)
        else:
            #take sorting based on phenotype independent component
            top_gene_or_pheno_inds = np.argsort(-(self.exp_pheno_factors if factor_gene_set_x_pheno else self.exp_gene_factors), axis=0)[:num_top,:]
            top_gene_set_inds = np.argsort(-self.exp_gene_set_factors, axis=0)[:num_top,:]


        self.factor_labels = []
        self.factor_top_gene_sets = []
        top_genes_or_phenos = []

        self.factor_anchor_top_gene_sets = []
        anchor_top_genes_or_phenos = []

        factor_prompts = []
        for i in range(self.num_factors()):
            self.factor_top_gene_sets.append([self.gene_sets[j] for j in top_gene_set_inds[:,i]])

            self.factor_anchor_top_gene_sets.append([[self.gene_sets[j] for j in top_anchor_gene_set_inds[:,i,k]] for k in range(top_anchor_gene_set_inds.shape[2])])

            if factor_gene_set_x_pheno:
                genes_or_phenos = self.phenos
            else:
                genes_or_phenos = self.genes

            top_genes_or_phenos.append([genes_or_phenos[j] for j in top_gene_or_pheno_inds[:,i]])

            anchor_top_genes_or_phenos.append([[genes_or_phenos[j] for j in top_anchor_gene_or_pheno_inds[:,i,k]] for k in range(top_anchor_gene_or_pheno_inds.shape[2])])

            self.factor_labels.append(self.factor_top_gene_sets[i][0] if len(self.factor_top_gene_sets[i]) > 0 else "")
            factor_prompts.append(",".join(self.factor_top_gene_sets[i]))

        if lmm_auth_key is not None and self.num_factors() > 0:
            if len(self.factor_top_gene_sets) == 1:
                prompt = "Print a label, five words maximum and no quotes, for: %s." % (" ".join(["%d. %s" % (j+1, ",".join(self.factor_top_gene_sets[j])) for j in range(len(self.factor_top_gene_sets))]))
            else:
                prompt = "Print a label, five words maximum, for each group. Print only labels, one per line, label number folowed by text: %s" % (" ".join(["%d. %s" % (j+1, ",".join(self.factor_top_gene_sets[j])) for j in range(len(self.factor_top_gene_sets))]))
            log("Querying LMM with prompt: %s" % prompt)
            response = query_lmm(prompt, lmm_auth_key)
            if response is not None:
                try:
                    responses = response.strip().split("\n")
                    responses = [x for x in responses if len(x) > 0]
                    if len(responses) == len(self.factor_labels):
                        for i in range(len(self.factor_labels)):
                            cur_response = responses[i]
                            cur_response_tokens = cur_response.split()
                            if len(cur_response_tokens) > 1 and cur_response_tokens[0][-1] == ".":
                                try:
                                    number = int(cur_response_tokens[0][:-1])
                                    cur_response = " ".join(cur_response_tokens[1:])
                                except ValueError:
                                    pass
                            self.factor_labels[i] = cur_response
                    else:
                        raise Exception
                except Exception:
                    log("Couldn't decode LMM response %s; using simple label" % response)
                    pass

        if factor_gene_set_x_pheno:
            self.factor_top_phenos = top_genes_or_phenos
            self.factor_anchor_top_phenos = anchor_top_genes_or_phenos
        else:
            self.factor_top_genes = top_genes_or_phenos
            self.factor_anchor_top_genes = anchor_top_genes_or_phenos

        log("Found %d factors" % self.num_factors(), INFO)


    def _sparse_correlation_with_dot_product_threshold(self, X_sparse, beta, dot_product_threshold=0.01, Y=None):
        """
        Compute the sparse correlation matrix of (X * beta + Y) with dot-product thresholding,
        mean adjustment, and normalization, for k beta vectors in parallel.

        Parameters:
        - X_sparse (scipy.sparse.csc_matrix): Sparse matrix X of shape (n, m).
        - beta (np.array): Dense array of shape (k, m) for k beta vectors.
        - dot_product_threshold (float): Threshold for absolute dot product values.
        - Y (np.array, optional): Dense array of shape (k, n) or None. Defaults to None.

        Returns:
        - scipy.sparse.csc_matrix: Sparse block diagonal correlation matrix for k beta vectors.
        """

        # Handle Y as an optional argument
        if Y is not None:
            if beta.shape[0] != Y.shape[0] or X_sparse.shape[0] != Y.shape[1]:
                raise ValueError("Y must have shape (k, n) where k matches beta's rows and n matches X's rows.")
            Y = np.square(Y.flatten())

        # Ensure beta is 2D
        if beta.ndim == 1:
            beta = beta[np.newaxis, :]  # Convert to shape (1, m) if beta is a single vector

        k, m = beta.shape  # Number of beta vectors and features
        n = X_sparse.shape[0]  # Number of rows (samples) in X

        # Step 1: Scale X_sparse by each beta vector and construct block diagonal matrix
        scaled_blocks = [X_sparse.multiply(beta[i, :]) for i in range(k)]
        X_scaled = sparse.block_diag(scaled_blocks, format='csc')  # Shape: (k * n, k * m)

        var_threshold = 0.05
        prior_threshold = 0.1
        X_scaled_sum = X_scaled.sum(axis=1).A1
        keep_mask = np.logical_and((np.square(X_scaled_sum) / ((Y if Y is not None else 0) + np.square(X_scaled_sum) + 1e-20) > var_threshold), (X_scaled_sum > prior_threshold))

        X_scaled = (X_scaled.T.multiply(keep_mask)).T
        X_scaled.eliminate_zeros()

        # Step 2: Compute uncentered second moment for all scaled X_sparse blocks

        X_scaled_dot_X_scaled = X_scaled.dot(X_scaled.T).multiply(1.0 / m).tocsr()  # n x n

        # Retain only the rows, columns, and values that pass the threshold
        threshold_mask = np.abs(X_scaled_dot_X_scaled.data) < (dot_product_threshold / m)
        X_scaled_dot_X_scaled.data[threshold_mask] = 0
        X_scaled_dot_X_scaled.eliminate_zeros()

        #We now have E[XBi*XBj]

        #calculate E[Xbi] and E2[Xbi]

        E_X_scaled = X_scaled.mean(axis=1).A1
        E2_X_scaled = X_scaled_dot_X_scaled.diagonal()

        # Identify block and local indices
        if type(X_scaled_dot_X_scaled) is not sparse.csr_matrix:
            X_scaled_dot_X_scaled = X_scaled_dot_X_scaled.tocsr()

        #get indices of columns
        rows = np.repeat(np.arange(len(X_scaled_dot_X_scaled.indptr) - 1), np.diff(X_scaled_dot_X_scaled.indptr))
        cols = X_scaled_dot_X_scaled.indices  # Directly use indices for rows

        #subtract E[betai]E[betaj]
        X_scaled_dot_X_scaled.data -= E_X_scaled[rows] * E_X_scaled[cols]
        if Y is not None:
            X_scaled_dot_X_scaled.data += Y[rows] * Y[cols]
        #divide by the variances
        X_scaled_dot_X_scaled.data /= (np.sqrt((E2_X_scaled[rows] - np.square(E_X_scaled)[rows] + np.square(Y[rows] if Y is not None else 0)) * (E2_X_scaled[cols] - np.square(E_X_scaled)[cols] + np.square(Y[cols] if Y is not None else 0))) + 1e-20)

        cor_threshold = 0.01
        X_scaled_dot_X_scaled.data[X_scaled_dot_X_scaled.data <= cor_threshold] = 0
        X_scaled_dot_X_scaled.eliminate_zeros()

        # Step 5: Construct sparse block diagonal correlation matrix
        X_scaled_dot_X_scaled = X_scaled_dot_X_scaled + sparse.diags(np.ones(k * n), format="csr")
        X_scaled_dot_X_scaled = X_scaled_dot_X_scaled.multiply(sparse.diags(1.0 / X_scaled_dot_X_scaled.diagonal(), format="csr"))
        sparse_corr_matrix = X_scaled_dot_X_scaled

        # Step 6: Return sparse correlation matrix or list of matrices
        if k == 1:
            return sparse_corr_matrix
        else:
            return [sparse_corr_matrix[i * n:(i + 1) * n, i * n:(i + 1) * n] for i in range(k)]

    def run_phewas(self, gene_phewas_bfs_in=None, gene_phewas_bfs_id_col=None, gene_phewas_bfs_pheno_col=None, gene_phewas_bfs_log_bf_col=None, gene_phewas_bfs_combined_col=None, gene_phewas_bfs_prior_col=None, run_for_factors=False, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, sparse_solution=False, sparse_frac_betas=None, batch_size=1500, min_gene_factor_weight=0, **kwargs):

        #require X matrix
        if gene_phewas_bfs_in is None and self.gene_pheno_Y is None and self.gene_pheno_combined_prior_Ys is None and self.gene_pheno_priors is None:
            bail("Require --gene-bfs-in or --gene-phewas-bfs-in with a column for log_bf/Y in this operation")

        if run_for_factors:
            if self.exp_gene_set_factors is None:
                warn("Cannot run factor phewas without gene factors; skipping")
                return

            log("Running factor phewas", INFO)
        else:
            if self.genes is None:
                warn("Cannot run phewas without X matrix; skipping")
                return
            if self.Y is None and self.combined_prior_Ys is None and self.priors is None:
                warn("Cannot run phewas without Y values; skipping")
                return

            log("Running phewas", INFO)

        #first get the list of phenotypes
        read_file = gene_phewas_bfs_in is not None

        id_col = None
        pheno_col = None
        bf_col = None
        combined_col = None
        prior_col = None

        if read_file:
            if self.phenos is not None:
                phenos = copy.copy(self.phenos)
                pheno_to_ind = copy.copy(self.pheno_to_ind)
            else:
                phenos = []
                pheno_to_ind = {}

            self.num_gene_phewas_filtered = 0
            with open_gz(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
                log("Fetching phenotypes to use", DEBUG)
                header_cols = gene_phewas_bfs_fh.readline().strip().split()
                if gene_phewas_bfs_id_col is None:
                    gene_phewas_bfs_id_col = "Gene"
                if gene_phewas_bfs_pheno_col is None:
                    gene_phewas_bfs_pheno_col = "Pheno"

                id_col = self._get_col(gene_phewas_bfs_id_col, header_cols)
                pheno_col = self._get_col(gene_phewas_bfs_pheno_col, header_cols)
                if gene_phewas_bfs_log_bf_col is not None:
                    bf_col = self._get_col(gene_phewas_bfs_log_bf_col, header_cols)
                else:
                    bf_col = self._get_col("log_bf", header_cols, False)

                if gene_phewas_bfs_combined_col is not None:
                    combined_col = self._get_col(gene_phewas_bfs_combined_col, header_cols, True)
                else:
                    combined_col = self._get_col("combined", header_cols, False)

                prior_col = None
                if gene_phewas_bfs_prior_col is not None:
                    prior_col = self._get_col(gene_phewas_bfs_prior_col, header_cols, True)
                else:
                    prior_col = self._get_col("prior", header_cols, False)

                for line in gene_phewas_bfs_fh:
                    cols = line.strip().split()
                    if id_col >= len(cols) or pheno_col >= len(cols) or (bf_col is not None and bf_col >= len(cols)) or (combined_col is not None and combined_col >= len(cols)) or (prior_col is not None and prior_col >= len(cols)):
                        warn("Skipping due to too few columns in line: %s" % line)
                        continue

                    gene = cols[id_col]
                    if self.gene_label_map is not None and gene in self.gene_label_map:
                        gene = self.gene_label_map[gene]

                    if gene not in self.gene_to_ind:
                        continue

                    pheno = cols[pheno_col]

                    if pheno not in pheno_to_ind:
                        pheno_to_ind[pheno] = len(phenos)
                        phenos.append(pheno)

                #update what's stored internally
                num_added_phenos = 0
                if self.phenos is not None and len(self.phenos) < len(phenos):
                    num_added_phenos = len(phenos) - len(self.phenos)

                if num_added_phenos > 0:
                    if self.X_phewas_beta is not None:
                        self.X_phewas_beta = sparse.csc_matrix(sparse.vstack((self.X_phewas_beta, sparse.csc_matrix((num_added_phenos, self.X_phewas_beta.shape[1])))))
                    if self.X_phewas_beta_uncorrected is not None:
                        self.X_phewas_beta_uncorrected = sparse.csc_matrix(sparse.vstack((self.X_phewas_beta_uncorrected, sparse.csc_matrix((num_added_phenos, self.X_phewas_beta_uncorrected.shape[1])))))
                    if self.gene_pheno_Y is not None:
                        self.gene_pheno_Y = sparse.csc_matrix(sparse.hstack((self.gene_pheno_Y, sparse.csc_matrix((self.gene_pheno_Y.shape[0], num_added_phenos)))))
                    if self.gene_pheno_combined_prior_Ys is not None:
                        self.gene_pheno_combined_prior_Ys = sparse.csc_matrix(sparse.hstack((self.gene_pheno_combined_prior_Ys, sparse.csc_matrix((self.gene_pheno_combined_prior_Ys.shape[0], num_added_phenos)))))
                    if self.gene_pheno_priors is not None:
                        self.gene_pheno_priors = sparse.csc_matrix(sparse.hstack((self.gene_pheno_priors, sparse.csc_matrix((self.gene_pheno_priors.shape[0], num_added_phenos)))))

                self.phenos = phenos
                pheno_to_ind = self._construct_map_to_ind(phenos)

        else:
            phenos = self.phenos

        #do phewas in batches to save memory
        num_batches = int(np.ceil(len(phenos) / batch_size))
        #always have three vectors even if some are None
        if run_for_factors:
            input_values = self.exp_gene_factors
            factor_keep_mask = np.full(input_values.shape[0], True)

            if min_gene_factor_weight > 0:
                factor_keep_mask = np.any(self.exp_gene_factors > min_gene_factor_weight, axis=1)

        else:
            default_value = self.Y[:,np.newaxis] if self.Y is not None else self.combined_prior_Ys[:,np.newaxis] if self.combined_prior_Ys is not None else self.priors[:,np.newaxis]

            input_values = np.hstack((self.Y[:,np.newaxis] if self.Y is not None else default_value, self.combined_prior_Ys[:,np.newaxis] if self.combined_prior_Ys is not None else default_value, self.priors[:,np.newaxis] if self.priors is not None else default_value))

            #convert these to probabilities
            input_values = np.exp(input_values + self.background_bf) / (1 + np.exp(input_values + self.background_bf))


        def _calculate_phewas(X_mat, Y_mat, X_orig=None, X_phewas_beta=None, Y_resid=None, multivariate=False, covs=None, huber=False):
            (mean_shifts, scale_factors) = self._calc_X_shift_scale(X_mat)

            cor_matrices = None

            beta_tildes = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
            ses = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
            z_scores = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
            p_values = np.zeros((Y_mat.shape[0], X_mat.shape[1]))
            se_inflation_factors = np.zeros((Y_mat.shape[0], X_mat.shape[1]))

            cor_batch_size = int(np.ceil(beta_tildes.shape[0] / 4) if X_phewas_beta is not None and X_orig is not None else beta_tildes.shape[0])

            num_cor_batches = int(np.ceil(beta_tildes.shape[0] / cor_batch_size))
            for batch in range(num_cor_batches):
                log("Processing block batch %s" % (batch), TRACE)
                begin = batch * cor_batch_size
                end = (batch + 1) * cor_batch_size
                if end > beta_tildes.shape[0]:
                    end = beta_tildes.shape[0]
                cur_batch_size = end - begin

                if X_phewas_beta is not None and X_orig is not None and not options.debug_skip_correlation:
                    if X_phewas_beta.shape[0] != Y_mat.shape[0]:
                        bail("When calling this, the phewas_betas must have same number of phenos as Y_mat: shapes are X_phewas=(%d,%d) vs. Y_mat=(%d,%d)" % (X_phewas_beta.shape[0], X_phewas_beta.shape[1], Y_mat.shape[0], Y_mat.shape[1]))
                    #require them to share at least one gene set with beta above 0.01
                    dot_threshold = 0.01 * 0.01
                    log("Calculating correlation matrix for use in residuals", DEBUG)
                    cor_matrices = self._sparse_correlation_with_dot_product_threshold(X_orig, X_phewas_beta[begin:end,:], dot_product_threshold=dot_threshold, Y=Y_resid[begin:end,:])

                    total = 0
                    nnz = 0
                    for cor_matrix in cor_matrices if type(cor_matrices) is list else [cor_matrices]:
                        total += np.prod(cor_matrix.shape)
                        nnz += cor_matrix.nnz
                    log("Sparsity of correlation matrix is %d/%d=%.3g (size %.3gMb)" % (nnz, total, float(nnz)/total, nnz * 8 / (1024 * 1024)), DEBUG)

                if multivariate:
                    if huber:
                        #(beta_tildes[begin:end,:], ses[begin:end,:], z_scores[begin:end,:], p_values[begin:end,:], se_inflation_factors[begin:end,:]) = self._compute_multivariate_beta_tildes_huber_correlated(X_mat, Y_mat[begin:end,:], resid_correlation_matrix=cor_matrices, covs=covs if not options.debug_skip_phewas_covs else None)

                        (beta_tildes[begin:end,:], ses[begin:end,:], z_scores[begin:end,:], p_values[begin:end,:], se_inflation_factors[begin:end,:]) = self._compute_robust_betas(X_mat, Y_mat[begin:end,:], resid_correlation_matrix=cor_matrices, covs=covs if not options.debug_skip_phewas_covs else None)                    
                    else:
                        (beta_tildes[begin:end,:], ses[begin:end,:], z_scores[begin:end,:], p_values[begin:end,:], se_inflation_factors[begin:end,:]) = self._compute_multivariate_beta_tildes(X_mat, Y_mat[begin:end,:], resid_correlation_matrix=cor_matrices, covs=covs if not options.debug_skip_phewas_covs else None)
                else:
                    (beta_tildes[begin:end,:], ses[begin:end,:], z_scores[begin:end,:], p_values[begin:end,:], se_inflation_factors[begin:end,:]) = self._compute_beta_tildes(X_mat, Y_mat[begin:end,:], scale_factors=scale_factors, mean_shifts=mean_shifts, resid_correlation_matrix=cor_matrices)


            one_sided_p_values = copy.copy(p_values)
            one_sided_p_values[z_scores < 0] = 1 - p_values[z_scores < 0] / 2.0
            one_sided_p_values[z_scores > 0] = p_values[z_scores > 0] / 2.0

            if multivariate:
                return (None, None, beta_tildes.T, ses.T, z_scores.T, p_values.T, one_sided_p_values.T)

            #now run the betas (no correlations here)
            #due to (bad) design of accessing sigma/p as member variables, we have to set and restore them surrounding the call
            #this will use internal sigma2 and p
            orig_ps = self.ps
            orig_sigma2s = self.sigma2s
            orig_p = self.p
            orig_sigma2 = self.sigma2
            orig_sigma_power = self.sigma_power
            #there are always none for running betas here
            self.ps = None
            self.sigma2s = None
            #self.p = 0.1
            #self.sigma2 = 0.01
            #self.sigma_power = orig_sigma_power
            new_p = 0.5
            new_sigma2 = orig_sigma2 * (new_p / orig_p)
            self.set_p(new_p)
            self.set_sigma(new_sigma2, orig_sigma_power, convert_sigma_to_internal_units=False)
            update_hyper_p = False
            update_hyper_sigma = False

            (betas_uncorrected, postp_uncorrected) = self._calculate_non_inf_betas(initial_p=self.p, assume_independent=True, beta_tildes=beta_tildes, ses=ses, V=None, X_orig=None, scale_factors=scale_factors, mean_shifts=mean_shifts, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, **kwargs)

            self.ps = orig_ps
            self.sigma2s = orig_sigma2s
            self.p = orig_p
            self.sigma2 = orig_sigma2
            self.sigma_power = orig_sigma_power

            return (betas_uncorrected / scale_factors).T, postp_uncorrected.T, (beta_tildes / scale_factors).T, (ses / scale_factors).T, z_scores.T, p_values.T, one_sided_p_values.T


        def __update_pheno_vec(self_beta, self_beta_tilde, self_se, self_Z, self_p_value, self_one_sided_p_value, beta, beta_tilde, se, Z, p_value, one_sided_p_value):
            if self_beta_tilde is None:
                return beta, beta_tilde, se, Z, p_value, one_sided_p_value
            else:
                beta_append = np.hstack((self_beta, beta)) if self_beta is not None else None
                return beta_append, np.hstack((self_beta_tilde, beta_tilde)), np.hstack((self_se, se)), np.hstack((self_Z, Z)), np.hstack((self_p_value, p_value)), np.hstack((self_one_sided_p_value, one_sided_p_value)) if self_one_sided_p_value is not None else None




        for batch in range(num_batches):
            log("Getting phenos block batch %s" % (batch), TRACE)

            begin = batch * batch_size
            end = (batch + 1) * batch_size
            if end > len(phenos):
                end = len(phenos)

            cur_batch_size = end - begin
            log("Processing phenos %d-%d" % (begin + 1, end))

            if read_file:
                gene_pheno_Y = np.zeros((len(self.genes), cur_batch_size)) if bf_col is not None else None
                gene_pheno_combined_prior_Ys = np.zeros((len(self.genes), cur_batch_size)) if combined_col is not None else None
                gene_pheno_priors = np.zeros((len(self.genes), cur_batch_size)) if prior_col is not None else None

                with open_gz(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
                    for line in gene_phewas_bfs_fh:
                        cols = line.strip().split()
                        if id_col >= len(cols) or pheno_col >= len(cols) or (bf_col is not None and bf_col >= len(cols)) or (combined_col is not None and combined_col >= len(cols)) or (prior_col is not None and prior_col >= len(cols)):
                            warn("Skipping due to too few columns in line: %s" % line)
                            continue

                        gene = cols[id_col]
                        if self.gene_label_map is not None and gene in self.gene_label_map:
                            gene = self.gene_label_map[gene]

                        if gene not in self.gene_to_ind:
                            continue

                        gene_ind = self.gene_to_ind[gene]

                        pheno = cols[pheno_col]

                        if pheno not in pheno_to_ind:
                            continue

                        pheno_ind = pheno_to_ind[pheno] - begin
                        if pheno_ind < 0 or pheno_ind >= cur_batch_size:
                            continue


                        if combined_col is not None:
                            try:
                                combined = float(cols[combined_col])
                            except ValueError:
                                if not cols[combined_col] == "NA":
                                    warn("Skipping unconvertible value %s for gene_set %s" % (cols[combined_col], gene))
                                continue

                            gene_pheno_combined_prior_Ys[gene_ind,pheno_ind] = combined

                        if bf_col is not None:
                            try:
                                bf = float(cols[bf_col])
                            except ValueError:
                                if not cols[bf_col] == "NA":
                                    warn("Skipping unconvertible value %s for gene %s and pheno %s" % (cols[bf_col], gene, pheno))
                                continue

                            gene_pheno_Y[gene_ind,pheno_ind] = bf

                        if prior_col is not None:
                            try:
                                prior = float(cols[prior_col])
                            except ValueError:
                                if not cols[prior_col] == "NA":
                                    warn("Skipping unconvertible value %s for gene_set %s" % (cols[prior_col], gene))
                                continue

                            gene_pheno_prior[gene_ind,pheno_ind] = prior

            else:
                gene_pheno_Y = self.gene_pheno_Y[:,begin:end].toarray() if self.gene_pheno_Y is not None else None
                gene_pheno_combined_prior_Ys = self.gene_pheno_combined_prior_Ys[:,begin:end].toarray() if self.gene_pheno_combined_prior_Ys is not None else None


            if run_for_factors:
                #in multivariate mode the returned beta tildes are actually betas
                if gene_pheno_Y is not None:
                    _, _, beta_tilde, se, Z, p_value, one_sided_p_value = _calculate_phewas(input_values[factor_keep_mask,:], gene_pheno_Y[factor_keep_mask,:].T, multivariate=True, covs=self.Y[factor_keep_mask])

                    _, self.factor_phewas_Y_betas, self.factor_phewas_Y_ses, self.factor_phewas_Y_zs, self.factor_phewas_Y_p_values, self.factor_phewas_Y_one_sided_p_values = __update_pheno_vec(None, self.factor_phewas_Y_betas, self.factor_phewas_Y_ses, self.factor_phewas_Y_zs, self.factor_phewas_Y_p_values, self.factor_phewas_Y_one_sided_p_values, None, beta_tilde, se, Z, p_value, one_sided_p_value)

                    if not options.debug_skip_huber:
                        _, _, beta_tilde, se, Z, p_value, one_sided_p_value = _calculate_phewas(input_values[factor_keep_mask,:], gene_pheno_Y[factor_keep_mask,:].T, multivariate=True, covs=self.Y[factor_keep_mask], huber=True)

                        _, self.factor_phewas_Y_huber_betas, self.factor_phewas_Y_huber_ses, self.factor_phewas_Y_huber_zs, self.factor_phewas_Y_huber_p_values, self.factor_phewas_Y_huber_one_sided_p_values = __update_pheno_vec(None, self.factor_phewas_Y_huber_betas, self.factor_phewas_Y_huber_ses, self.factor_phewas_Y_huber_zs, self.factor_phewas_Y_huber_p_values, self.factor_phewas_Y_huber_one_sided_p_values, None, beta_tilde, se, Z, p_value, one_sided_p_value)

                if gene_pheno_combined_prior_Ys is not None and not options.debug_skip_correlation:

                    _, _, beta_tilde, se, Z, p_value, one_sided_p_value = _calculate_phewas(input_values[factor_keep_mask,:], gene_pheno_combined_prior_Ys[factor_keep_mask,:].T, X_orig=self.X_orig[factor_keep_mask,:], X_phewas_beta=self.X_phewas_beta[begin:end,:] if self.X_phewas_beta is not None else None, Y_resid=gene_pheno_Y[factor_keep_mask,:].T, multivariate=True, covs=self.combined_prior_Ys[factor_keep_mask] if self.combined_prior_Ys is not None else self.Y[factor_keep_mask])

                    _, self.factor_phewas_combined_prior_Ys_betas, self.factor_phewas_combined_prior_Ys_ses, self.factor_phewas_combined_prior_Ys_zs, self.factor_phewas_combined_prior_Ys_p_values, self.factor_phewas_combined_prior_Ys_one_sided_p_values = __update_pheno_vec(None, self.factor_phewas_combined_prior_Ys_betas, self.factor_phewas_combined_prior_Ys_ses, self.factor_phewas_combined_prior_Ys_zs, self.factor_phewas_combined_prior_Ys_p_values, self.factor_phewas_combined_prior_Ys_one_sided_p_values, None, beta_tilde, se, Z, p_value, one_sided_p_value)

                    if not options.debug_skip_huber:

                        _, _, beta_tilde, se, Z, p_value, one_sided_p_value = _calculate_phewas(input_values[factor_keep_mask,:], gene_pheno_combined_prior_Ys[factor_keep_mask,:].T, X_orig=self.X_orig[factor_keep_mask,:], X_phewas_beta=self.X_phewas_beta[begin:end,:] if self.X_phewas_beta is not None else None, Y_resid=gene_pheno_Y[factor_keep_mask,:].T, multivariate=True, covs=self.combined_prior_Ys[factor_keep_mask] if self.combined_prior_Ys is not None else self.Y[factor_keep_mask], huber=True)

                        _, self.factor_phewas_combined_prior_Ys_huber_betas, self.factor_phewas_combined_prior_Ys_huber_ses, self.factor_phewas_combined_prior_Ys_huber_zs, self.factor_phewas_combined_prior_Ys_huber_p_values, self.factor_phewas_combined_prior_Ys_huber_one_sided_p_values = __update_pheno_vec(None, self.factor_phewas_combined_prior_Ys_huber_betas, self.factor_phewas_combined_prior_Ys_huber_ses, self.factor_phewas_combined_prior_Ys_huber_zs, self.factor_phewas_combined_prior_Ys_huber_p_values, self.factor_phewas_combined_prior_Ys_huber_one_sided_p_values, None, beta_tilde, se, Z, p_value, one_sided_p_value)
            else:
                if gene_pheno_Y is not None:
                    beta, _, beta_tilde, se, Z, p_value, _ = _calculate_phewas(input_values, gene_pheno_Y.T)
                    assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
                    if self.Y is not None:
                        self.pheno_Y_vs_input_Y_beta, self.pheno_Y_vs_input_Y_beta_tilde, self.pheno_Y_vs_input_Y_se, self.pheno_Y_vs_input_Y_Z, self.pheno_Y_vs_input_Y_p_value, _ = __update_pheno_vec(self.pheno_Y_vs_input_Y_beta, self.pheno_Y_vs_input_Y_beta_tilde, self.pheno_Y_vs_input_Y_se, self.pheno_Y_vs_input_Y_Z, self.pheno_Y_vs_input_Y_p_value, None, beta[0,:], beta_tilde[0,:], se[0,:], Z[0,:], p_value[0,:], None)

                    if self.combined_prior_Ys is not None:
                        self.pheno_Y_vs_input_combined_prior_Ys_beta, self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde, self.pheno_Y_vs_input_combined_prior_Ys_se, self.pheno_Y_vs_input_combined_prior_Ys_Z, self.pheno_Y_vs_input_combined_prior_Ys_p_value, _ = __update_pheno_vec(self.pheno_Y_vs_input_combined_prior_Ys_beta, self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde, self.pheno_Y_vs_input_combined_prior_Ys_se, self.pheno_Y_vs_input_combined_prior_Ys_Z, self.pheno_Y_vs_input_combined_prior_Ys_p_value, None, beta[1,:], beta_tilde[1,:], se[1,:], Z[1,:], p_value[1,:], None)

                    if self.priors is not None:
                        self.pheno_Y_vs_input_priors_beta, self.pheno_Y_vs_input_priors_beta_tilde, self.pheno_Y_vs_input_priors_se, self.pheno_Y_vs_input_priors_Z, self.pheno_Y_vs_input_priors_p_value, _ = __update_pheno_vec(self.pheno_Y_vs_input_priors_beta, self.pheno_Y_vs_input_priors_beta_tilde, self.pheno_Y_vs_input_priors_se, self.pheno_Y_vs_input_priors_Z, self.pheno_Y_vs_input_priors_p_value, None, beta[2,:], beta_tilde[2,:], se[2,:], Z[2,:], p_value[2,:], None)

                if gene_pheno_combined_prior_Ys is not None and not options.debug_skip_correlation:
                    #we have to use the correlations here
                    beta, _, beta_tilde, se, Z, p_value, _ = _calculate_phewas(input_values, gene_pheno_combined_prior_Ys.T, X_orig=self.X_orig, X_phewas_beta=self.X_phewas_beta[begin:end,:] if self.X_phewas_beta is not None else None, Y_resid=gene_pheno_Y.T)
                    assert beta.shape[0] == 3, "First dimension of beta should be 3, not (%s, %s)" % (beta.shape[0], beta.shape[1])
                    if self.Y is not None:
                        self.pheno_combined_prior_Ys_vs_input_Y_beta, self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde, self.pheno_combined_prior_Ys_vs_input_Y_se, self.pheno_combined_prior_Ys_vs_input_Y_Z, self.pheno_combined_prior_Ys_vs_input_Y_p_value, _ = __update_pheno_vec(self.pheno_combined_prior_Ys_vs_input_Y_beta, self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde, self.pheno_combined_prior_Ys_vs_input_Y_se, self.pheno_combined_prior_Ys_vs_input_Y_Z, self.pheno_combined_prior_Ys_vs_input_Y_p_value, None, beta[0,:], beta_tilde[0,:], se[0,:], Z[0,:], p_value[0,:], None)

                    if self.combined_prior_Ys is not None:
                        self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value, _ = __update_pheno_vec(self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z, self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value, None, beta[1,:], beta_tilde[1,:], se[1,:], Z[1,:], p_value[1,:], None)


                    if self.priors is not None:
                        self.pheno_combined_prior_Ys_vs_input_priors_beta, self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde, self.pheno_combined_prior_Ys_vs_input_priors_se, self.pheno_combined_prior_Ys_vs_input_priors_Z, self.pheno_combined_prior_Ys_vs_input_priors_p_value, _ = __update_pheno_vec(self.pheno_combined_prior_Ys_vs_input_priors_beta, self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde, self.pheno_combined_prior_Ys_vs_input_priors_se, self.pheno_combined_prior_Ys_vs_input_priors_Z, self.pheno_combined_prior_Ys_vs_input_priors_p_value, None, beta[2,:], beta_tilde[2,:], se[2,:], Z[2,:], p_value[2,:], None)

    def run_sim(self, sigma2, p, sigma_power, log_bf_noise_sigma_mult=0, treat_sigma2_as_sigma2_cond=True):

        if sigma2 is None or sigma2 <= 0:
            bail("Require positive --sigma2 for simulations")
        if p is None:
            bail("Require --p-noninf for simulations")
        if sigma_power is None:
            bail("Require --sigma-power for simulations")
        if self.X_orig is None:
            bail("Require --X-in for simulations")
        
        log("Simulating gene set and gene values")
        #first simulate the sigmas
        self.betas = np.zeros(len(self.gene_sets))
        non_zero_gene_sets = np.random.random(self.betas.shape) < p

        scaled_sigma2s = self.get_scaled_sigma2(self.scale_factors, sigma2, sigma_power)

        #since we are only simulating for those that have non-zeros, we need to use conditional sigma2
        if treat_sigma2_as_sigma2_cond:
            sigma2_conds = scaled_sigma2s[non_zero_gene_sets]
            log("Using p=%.3g, sigma2_cond=%.3g" % (p, sigma2))
        else:
            sigma2_conds = scaled_sigma2s[non_zero_gene_sets] / p
            log("Using p=%.3g, sigma2_cond=%.3g" % (p, sigma2/p))

        self.betas[non_zero_gene_sets] = scipy.stats.norm.rvs(0, np.sqrt(sigma2_conds), np.sum(non_zero_gene_sets)).ravel()

        #now simulate the gene values
        self.priors = self.X_orig.dot(self.betas / self.scale_factors)

        if log_bf_noise_sigma_mult > 0:
            #here we don't divide by p since we are adding noise to every beta, not just non zero ones
            noise_add_betas = scipy.stats.norm.rvs(0, np.sqrt(scaled_sigma2s * log_bf_noise_sigma_mult), self.betas.shape)
            self.Y = self.priors + self.X_orig.dot(noise_add_betas / self.scale_factors)
        else:
            self.Y = self.priors

        self._set_Y(self.Y, self.Y, self.Y_exomes, self.Y_positive_controls)


    def get_col_sums(self, X, num_nonzero=False, axis=0):
        if num_nonzero:
            return X.astype(bool).sum(axis=axis).A1
        else:
            return np.abs(X).sum(axis=axis).A1

    def get_gene_N(self, get_missing=False):
        if get_missing:
            if self.gene_N_missing is None:
                return None
            else:
                return self.gene_N_missing + (self.gene_ignored_N_missing if self.gene_ignored_N_missing is not None else 0)
        else:
            if self.gene_N is None:
                return None
            else:
                return self.gene_N + (self.gene_ignored_N if self.gene_ignored_N is not None else 0)

    def write_gene_set_statistics(self, output_file, max_no_write_gene_set_beta=None, max_no_write_gene_set_beta_uncorrected=None, basic=False):
        log("Writing gene set stats to %s" % output_file, INFO)
        with open_gz(output_file, 'w') as output_fh:
            if self.gene_sets is None:
                return
            header = "Gene_Set"
            if self.gene_set_labels is not None:
                header = "%s\t%s" % (header, "label")
            if self.X_orig is not None:
                col_sums = self.get_col_sums(self.X_orig)
                header = "%s\t%s" % (header, "N")
                header = "%s\t%s" % (header, "scale")
            if self.beta_tildes is not None:
                header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde", "beta_tilde_internal", "P", "Z", "SE")
            if self.inf_betas is not None and not basic:
                header = "%s\t%s" % (header, "inf_beta")            
            if self.betas is not None:
                header = "%s\t%s\t%s" % (header, "beta", "beta_internal")
            if self.betas_uncorrected is not None and not basic:
                header = "%s\t%s" % (header, "beta_uncorrected")            
            if not basic:
                if self.non_inf_avg_cond_betas is not None:
                    header = "%s\t%s" % (header, "avg_cond_beta")            
                if self.non_inf_avg_postps is not None:
                    header = "%s\t%s" % (header, "avg_postp")            
                if self.beta_tildes_orig is not None:
                    header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde_orig", "beta_tilde_internal_orig", "P_orig", "Z_orig", "SE_orig")
                if self.inf_betas_orig is not None:
                    header = "%s\t%s" % (header, "inf_beta_orig")            
                if self.betas_orig is not None:
                    header = "%s\t%s\t%s" % (header, "beta_orig", "beta_internal_orig")
                if self.betas_uncorrected_orig is not None:
                    header = "%s\t%s\t%s" % (header, "beta_uncorrected_orig", "beta_uncorrected_internal_orig")
                if self.non_inf_avg_cond_betas_orig is not None:
                    header = "%s\t%s" % (header, "avg_cond_beta_orig")            
                if self.non_inf_avg_postps_orig is not None:
                    header = "%s\t%s" % (header, "avg_postp_orig")            
                if self.ps is not None or self.p is not None:
                    header = "%s\t%s" % (header, "p_used")
                if self.sigma2s is not None or self.sigma2 is not None:
                    header = "%s\t%s" % (header, "sigma2_used")
                if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                    header = "%s\t%s" % (header, "sigma2_thresholded")
                if self.X_osc is not None:
                    header = "%s\t%s\t%s\t%s" % (header, "O", "X_O", "weight")
                if self.total_qc_metrics is not None:
                    if options.debug_only_avg_huge:
                        header = "%s\t%s" % (header, "avg_huge_adjustment")
                    else:
                        header = "%s\t%s\t%s" % (header, "\t".join(map(lambda x: "avg_%s" % x, [self.gene_covariate_names[i] for i in range(len(self.gene_covariate_names)) if i != self.gene_covariate_intercept_index])), "avg_huge_adjustment")

                if self.mean_qc_metrics is not None:
                    header = "%s\t%s" % (header, "avg_avg_metric")

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.gene_sets))
            if self.betas is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.betas[k] / self.scale_factors[k])
            elif self.p_values is not None:
                ordered_i = sorted(ordered_i, key=lambda k: self.p_values[k])

            for i in ordered_i:

                if max_no_write_gene_set_beta is not None and self.betas is not None and np.abs(self.betas[i] / self.scale_factors[i]) <= max_no_write_gene_set_beta:
                    continue

                if max_no_write_gene_set_beta_uncorrected is not None and self.betas_uncorrected is not None and np.abs(self.betas_uncorrected[i] / self.scale_factors[i]) <= max_no_write_gene_set_beta_uncorrected:
                    continue

                line = self.gene_sets[i]
                if self.gene_set_labels is not None:
                    line = "%s\t%s" % (line, self.gene_set_labels[i])
                if self.X_orig is not None:
                    line = "%s\t%d" % (line, col_sums[i])
                    line = "%s\t%.3g" % (line, self.scale_factors[i])

                if self.beta_tildes is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes[i] / self.scale_factors[i], self.beta_tildes[i], self.p_values[i], self.z_scores[i], self.ses[i] / self.scale_factors[i])
                if self.inf_betas is not None and not basic:
                    line = "%s\t%.3g" % (line, self.inf_betas[i] / self.scale_factors[i])            
                if self.betas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, self.betas[i] / self.scale_factors[i], self.betas[i])
                if self.betas_uncorrected is not None and not basic:
                    line = "%s\t%.3g" % (line, self.betas_uncorrected[i] / self.scale_factors[i])            
                if not basic:
                    if self.non_inf_avg_cond_betas is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas[i] / self.scale_factors[i])
                    if self.non_inf_avg_postps is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_postps[i])
                    if self.beta_tildes_orig is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_orig[i] / self.scale_factors[i], self.beta_tildes_orig[i], self.p_values_orig[i], self.z_scores_orig[i], self.ses_orig[i] / self.scale_factors[i])
                    if self.inf_betas_orig is not None:
                        line = "%s\t%.3g" % (line, self.inf_betas_orig[i] / self.scale_factors[i])            
                    if self.betas_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, self.betas_orig[i] / self.scale_factors[i], self.betas_orig[i])
                    if self.betas_uncorrected_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, self.betas_uncorrected_orig[i] / self.scale_factors[i], self.betas_uncorrected_orig[i])
                    if self.non_inf_avg_cond_betas_orig is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas_orig[i] / self.scale_factors[i])
                    if self.non_inf_avg_postps_orig is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_postps_orig[i])

                    if self.ps is not None or self.p is not None:
                        line = "%s\t%.3g" % (line, self.ps[i] if self.ps is not None else self.p)
                    if self.sigma2s is not None or self.sigma2 is not None:
                        line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors[i], self.sigma2s[i] if self.sigma2s is not None else self.sigma2, self.sigma_power, None, None))
                    if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                        line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors[i], self.sigma2s[i] if self.sigma2s is not None else self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo))
                    if self.X_osc is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g" % (line, self.osc[i], self.X_osc[i], self.osc_weights[i])

                    if self.total_qc_metrics is not None:
                        line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, self.total_qc_metrics[i,:])))
                    if self.mean_qc_metrics is not None:
                        line = "%s\t%.3g" % (line, self.mean_qc_metrics[i])


                output_fh.write("%s\n" % line)

            if self.gene_sets_missing is not None:
                ordered_i = range(len(self.gene_sets_missing))
                if self.betas_missing is not None:
                    ordered_i = sorted(ordered_i, key=lambda k: -self.betas_missing[k] / self.scale_factors_missing[k])
                elif self.p_values_missing is not None:
                    ordered_i = sorted(ordered_i, key=lambda k: self.p_values_missing[k])

                col_sums_missing = self.get_col_sums(self.X_orig_missing_gene_sets)
                for i in range(len(self.gene_sets_missing)):
                    if max_no_write_gene_set_beta is not None and self.betas_missing is not None and np.abs(self.betas_missing[i] / self.scale_factors_missing[i]) <= max_no_write_gene_set_beta:
                        continue

                    if max_no_write_gene_set_beta_uncorrected is not None and self.betas_uncorrected_missing is not None and np.abs(self.betas_uncorrected_missing[i] / self.scale_factors_missing[i]) <= max_no_write_gene_set_beta_uncorrected:
                        continue

                    line = self.gene_sets_missing[i]
                    if self.gene_set_labels is not None:
                        line = "%s\t%s" % (line, self.gene_set_labels_missing[i])
                    line = "%s\t%d" % (line, col_sums_missing[i])
                    line = "%s\t%.3g" % (line, self.scale_factors_missing[i])

                    if self.beta_tildes is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_missing[i] / self.scale_factors_missing[i], self.beta_tildes_missing[i], self.p_values_missing[i], self.z_scores_missing[i], self.ses_missing[i] / self.scale_factors_missing[i])
                    if self.inf_betas is not None and not basic:
                        line = "%s\t%.3g" % (line, self.inf_betas_missing[i] / self.scale_factors_missing[i])            
                    if self.betas is not None:
                        line = "%s\t%.3g\t%.3g" % (line, self.betas_missing[i] / self.scale_factors_missing[i], self.betas_missing[i])
                    if self.betas_uncorrected is not None and not basic:
                        line = "%s\t%.3g" % (line, self.betas_uncorrected_missing[i] / self.scale_factors_missing[i])            
                    if not basic:
                        if self.non_inf_avg_cond_betas is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas_missing[i] / self.scale_factors_missing[i])
                        if self.non_inf_avg_postps is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_postps_missing[i])
                        if self.beta_tildes_orig is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_missing_orig[i] / self.scale_factors_missing[i], self.beta_tildes_missing_orig[i], self.p_values_missing_orig[i], self.z_scores_missing_orig[i], self.ses_missing_orig[i] / self.scale_factors_missing[i])
                        if self.inf_betas_orig is not None:
                            line = "%s\t%.3g" % (line, self.inf_betas_missing_orig[i] / self.scale_factors_missing[i])            
                        if self.betas_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, self.betas_missing_orig[i] / self.scale_factors_missing[i], self.betas_missing_orig[i])
                        if self.betas_uncorrected_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, self.betas_uncorrected_missing_orig[i] / self.scale_factors_missing[i], self.betas_uncorrected_missing_orig[i])
                        if self.non_inf_avg_cond_betas_orig is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas_missing_orig[i] / self.scale_factors_missing[i])
                        if self.non_inf_avg_postps_orig is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_postps_missing_orig[i])

                        if self.ps is not None or self.p is not None:
                            line = "%s\t%.3g" % (line, self.ps_missing[i] if self.ps_missing is not None else self.p)

                        if self.sigma2s is not None or self.sigma2 is not None:
                            line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors_missing[i], self.sigma2s_missing[i] if self.sigma2s_missing is not None else self.sigma2, self.sigma_power, None, None))
                        if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                            line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors_missing[i], self.sigma2s_missing[i] if self.sigma2s_missing is not None else self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo))

                        if self.X_osc is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g" % (line, self.osc_missing[i], self.X_osc_missing[i], self.osc_weights_missing[i])

                        if self.total_qc_metrics is not None:
                            line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, self.total_qc_metrics_missing[i,:])))
                        if self.mean_qc_metrics is not None:
                            line = "%s\t%.3g" % (line, self.mean_qc_metrics_missing[i])

                    output_fh.write("%s\n" % line)



            if self.gene_sets_ignored is not None:

                ordered_i = range(len(self.gene_sets_ignored))
                if self.p_values_ignored is not None:
                    ordered_i = sorted(ordered_i, key=lambda k: self.p_values_ignored[k])

                for i in ordered_i:
                    ignored_beta_value = 0 
                    if max_no_write_gene_set_beta is not None and self.betas is not None and ignored_beta_value <= max_no_write_gene_set_beta:
                        continue

                    ignored_beta_uncorrected_value = 0 
                    if max_no_write_gene_set_beta_uncorrected is not None and self.betas_uncorrected is not None and ignored_beta_uncorrected_value <= max_no_write_gene_set_beta_uncorrected:
                        continue


                    line = "%s" % self.gene_sets_ignored[i]
                    if self.gene_set_labels is not None:
                        line = "%s\t%s" % (line, self.gene_set_labels_ignored[i])

                    line = "%s\t%d" % (line, self.col_sums_ignored[i])
                    line = "%s\t%.3g" % (line, self.scale_factors_ignored[i])

                    scale_factor_denom = self.scale_factors_ignored[i] + 1e-20

                    if self.beta_tildes is not None:
                        if self.beta_tildes_ignored is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_ignored[i] / scale_factor_denom, self.beta_tildes_ignored[i], self.p_values_ignored[i], self.z_scores_ignored[i], self.ses_ignored[i] / scale_factor_denom)
                        else:
                            line = "%s\t%s\t%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA", "NA", "NA")
                    if self.inf_betas is not None and not basic:
                        line = "%s\t%.3g" % (line, 0)            
                    if self.betas is not None:
                        line = "%s\t%.3g\t%.3g" % (line, ignored_beta_value, ignored_beta_value)
                    if self.betas_uncorrected is not None and not basic:
                        line = "%s\t%.3g" % (line, ignored_beta_uncorrected_value)            
                    if not basic:
                        if self.non_inf_avg_cond_betas is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.non_inf_avg_postps is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.beta_tildes_orig is not None:
                            if self.beta_tildes_ignored is not None:
                                line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_ignored[i] / scale_factor_denom, self.beta_tildes_ignored[i], self.p_values_ignored[i], self.z_scores_ignored[i], self.ses_ignored[i] / scale_factor_denom)
                            else:
                                line = "%s\t%s\t%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA", "NA", "NA")
                        if self.inf_betas_orig is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.betas_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                        if self.betas_uncorrected_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                        if self.non_inf_avg_cond_betas_orig is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.non_inf_avg_postps_orig is not None:
                            line = "%s\t%.3g" % (line, 0)

                        if self.ps is not None or self.p is not None:
                            line = "%s\t%s" % (line, "NA")
                        if self.sigma2s is not None or self.sigma2 is not None:
                            line = "%s\t%s" % (line, "NA")
                        if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                            line = "%s\t%s" % (line, "NA")

                        if self.X_osc is not None:
                            line = "%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA")

                        if self.total_qc_metrics is not None:
                            line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, self.total_qc_metrics_ignored[i,:])))
                        if self.mean_qc_metrics is not None:
                            line = "%s\t%.3g" % (line, self.mean_qc_metrics_ignored[i])

                    output_fh.write("%s\n" % line)

    def write_gene_statistics(self, output_file):
        log("Writing gene stats to %s" % output_file, INFO)

        with open_gz(output_file, 'w') as output_fh:
            if self.genes is not None:
                genes = self.genes
            elif self.gene_to_huge_score is not None:
                genes = list(self.gene_to_huge_score.keys())
            elif self.gene_to_gwas_huge_score is not None:
                genes = list(self.gene_to_huge_score.keys())
            elif self.gene_to_huge_score is not None:
                genes = list(self.gene_to_huge_score.keys())
            else:
                return

            huge_only_genes = set()
            if self.gene_to_huge_score is not None:
                huge_only_genes = set(self.gene_to_huge_score.keys()) - set(genes)
            if self.gene_to_gwas_huge_score is not None:
                huge_only_genes = set(self.gene_to_gwas_huge_score.keys()) - set(genes) - set(huge_only_genes)
            if self.gene_to_exomes_huge_score is not None:
                huge_only_genes = set(self.gene_to_exomes_huge_score.keys()) - set(genes) - set(huge_only_genes)

            if self.genes_missing is not None:
                huge_only_genes = huge_only_genes - set(self.genes_missing)

            huge_only_genes = list(huge_only_genes)

            write_regression = self.Y_for_regression is not None and self.Y is not None and np.any(~np.isclose(self.Y, self.Y_for_regression))

            header = "Gene"

            if self.priors is not None:
                header = "%s\t%s" % (header, "prior")
            if self.priors_adj is not None:
                header = "%s\t%s" % (header, "prior_adj")
            if self.combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "combined")
            if self.combined_prior_Ys_adj is not None:
                header = "%s\t%s" % (header, "combined_adj")
            if self.combined_prior_Y_ses is not None:
                header = "%s\t%s" % (header, "combined_se")
            if self.combined_Ds is not None:
                header = "%s\t%s" % (header, "combined_D")
            if self.gene_to_huge_score is not None:
                header = "%s\t%s" % (header, "huge_score")
            if self.gene_to_gwas_huge_score is not None:
                header = "%s\t%s" % (header, "huge_score_gwas")
            if self.gene_to_gwas_huge_score_uncorrected is not None:
                header = "%s\t%s" % (header, "huge_score_gwas_uncorrected")
            if self.gene_to_exomes_huge_score is not None:
                header = "%s\t%s" % (header, "huge_score_exomes")
            if self.gene_to_positive_controls is not None:
                header = "%s\t%s" % (header, "positive_control")
            if self.Y is not None:
                header = "%s\t%s" % (header, "log_bf")
            if write_regression:
                header = "%s\t%s" % (header, "log_bf_regression")
            if self.Y_uncorrected is not None:
                header = "%s\t%s" % (header, "log_bf_uncorrected")
            if self.Y_w is not None:
                header = "%s\t%s" % (header, "log_bf_w")
            if self.Y_fw is not None:
                header = "%s\t%s" % (header, "log_bf_fw")
            if self.priors_orig is not None:
                header = "%s\t%s" % (header, "prior_orig")
            if self.priors_adj_orig is not None:
                header = "%s\t%s" % (header, "prior_adj_orig")
            if self.batches is not None:
                header = "%s\t%s" % (header, "batch")
            if self.X_orig is not None:
                header = "%s\t%s" % (header, "N")            
            if self.gene_to_chrom is not None:
                header = "%s\t%s" % (header, "Chrom")
            if self.gene_to_pos is not None:
                header = "%s\t%s\t%s" % (header, "Start", "End")

            if self.gene_covariate_zs is not None:
                header = "%s\t%s" % (header, "\t".join(map(lambda x: "%s" % x, [self.gene_covariate_names[i] for i in range(len(self.gene_covariate_names)) if i != self.gene_covariate_intercept_index])))

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.genes))
            if self.combined_prior_Ys is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.combined_prior_Ys[k])
            elif self.priors is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.priors[k])
            elif self.Y is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y[k])
            elif write_regression:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y_for_regression[k])

            gene_N = self.get_gene_N()
            for i in ordered_i:
                gene = genes[i]
                line = gene
                if self.priors is not None:
                    line = "%s\t%.3g" % (line, self.priors[i])
                if self.priors_adj is not None:
                    line = "%s\t%.3g" % (line, self.priors_adj[i])
                if self.combined_prior_Ys is not None:
                    line = "%s\t%.3g" % (line, self.combined_prior_Ys[i])
                if self.combined_prior_Ys_adj is not None:
                    line = "%s\t%.3g" % (line, self.combined_prior_Ys_adj[i])
                if self.combined_prior_Y_ses is not None:
                    line = "%s\t%.3g" % (line, self.combined_prior_Y_ses[i])
                if self.combined_Ds is not None:
                    line = "%s\t%.3g" % (line, self.combined_Ds[i])
                if self.gene_to_huge_score is not None:
                    if gene in self.gene_to_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score is not None:
                    if gene in self.gene_to_gwas_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score_uncorrected is not None:
                    if gene in self.gene_to_gwas_huge_score_uncorrected:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score_uncorrected[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_exomes_huge_score is not None:
                    if gene in self.gene_to_exomes_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_exomes_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_positive_controls is not None:
                    if gene in self.gene_to_positive_controls:
                        line = "%s\t%.3g" % (line, self.gene_to_positive_controls[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.Y is not None:
                    line = "%s\t%.3g" % (line, self.Y[i])
                if write_regression:
                    line = "%s\t%.3g" % (line, self.Y_for_regression[i])
                if self.Y_uncorrected is not None:
                    line = "%s\t%.3g" % (line, self.Y_uncorrected[i])
                if self.Y_w is not None:
                    line = "%s\t%.3g" % (line, self.Y_w[i])
                if self.Y_fw is not None:
                    line = "%s\t%.3g" % (line, self.Y_fw[i])
                if self.priors_orig is not None:
                    line = "%s\t%.3g" % (line, self.priors_orig[i])
                if self.priors_adj_orig is not None:
                    line = "%s\t%.3g" % (line, self.priors_adj_orig[i])
                if self.batches is not None:
                    line = "%s\t%s" % (line, self.batches[i])
                if self.X_orig is not None:
                    line = "%s\t%d" % (line, gene_N[i])
                if self.gene_to_chrom is not None:
                    line = "%s\t%s" % (line, self.gene_to_chrom[gene] if gene in self.gene_to_chrom else "NA")
                if self.gene_to_pos is not None:
                    line = "%s\t%s\t%s" % (line, self.gene_to_pos[gene][0] if gene in self.gene_to_pos else "NA", self.gene_to_pos[gene][1] if gene in self.gene_to_pos else "NA")

                if self.gene_covariate_zs is not None:
                    line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, [self.gene_covariate_zs[i,j] for j in range(len(self.gene_covariate_names)) if j != self.gene_covariate_intercept_index])))

                output_fh.write("%s\n" % line)

            if self.genes_missing is not None:
                gene_N_missing = self.get_gene_N(get_missing=True)

                for i in range(len(self.genes_missing)):
                    gene = self.genes_missing[i]
                    line = gene
                    if self.priors is not None:
                        line = ("%s\t%.3g" % (line, self.priors_missing[i])) if self.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.priors_adj is not None:
                        line = ("%s\t%.3g" % (line, self.priors_adj_missing[i])) if self.priors_adj_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.combined_prior_Ys is not None:
                        #has no Y of itself so its combined is just the prior
                        line = ("%s\t%.3g" % (line, self.priors_missing[i])) if self.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.combined_prior_Ys_adj is not None:
                        #has no Y of itself so its combined is just the prior
                        line = ("%s\t%.3g" % (line, self.priors_adj_missing[i])) if self.priors_adj_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.combined_prior_Y_ses is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.combined_Ds_missing is not None:
                        line = "%s\t%.3g" % (line, self.combined_Ds_missing[i])
                    if self.gene_to_huge_score is not None:
                        if gene in self.gene_to_huge_score:
                            line = "%s\t%.3g" % (line, self.gene_to_huge_score[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_gwas_huge_score is not None:
                        if gene in self.gene_to_gwas_huge_score:
                            line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_gwas_huge_score_uncorrected is not None:
                        if gene in self.gene_to_gwas_huge_score_uncorrected:
                            line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score_uncorrected[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_exomes_huge_score is not None:
                        if gene in self.gene_to_exomes_huge_score:
                            line = "%s\t%.3g" % (line, self.gene_to_exomes_huge_score[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_positive_controls is not None:
                        if gene in self.gene_to_positive_controls:
                            line = "%s\t%.3g" % (line, self.gene_to_positive_controls[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.Y is not None:
                        line = "%s\t%s" % (line, "NA")
                    if write_regression:
                        line = "%s\t%s" % (line, "NA")
                    if self.Y_uncorrected is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.Y_w is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.Y_fw is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.priors_orig is not None:
                        line = ("%s\t%.3g" % (line, self.priors_missing_orig[i])) if self.priors_missing_orig is not None else ("%s\t%s" % (line, "NA"))

                    if self.priors_adj_orig is not None:
                        line = ("%s\t%.3g" % (line, self.priors_adj_missing_orig[i])) if self.priors_adj_missing_orig is not None else ("%s\t%s" % (line, "NA"))
                    if self.batches is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.X_orig is not None:
                        line = "%s\t%d" % (line, gene_N_missing[i])
                    if self.gene_to_chrom is not None:
                        line = "%s\t%s" % (line, self.gene_to_chrom[gene] if gene in self.gene_to_chrom else "NA")
                    if self.gene_to_pos is not None:
                        line = "%s\t%s\t%s" % (line, self.gene_to_pos[gene][0] if gene in self.gene_to_pos else "NA", self.gene_to_pos[gene][1] if gene in self.gene_to_pos else "NA")

                    if self.gene_covariate_zs is not None:
                        line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(self.gene_covariate_names)) if j != self.gene_covariate_intercept_index]))

                    output_fh.write("%s\n" % line)

            for i in range(len(huge_only_genes)):
                gene = huge_only_genes[i]
                line = gene
                if self.priors is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.priors_adj is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_prior_Ys is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_prior_Ys_adj is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_prior_Y_ses is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_Ds_missing is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.gene_to_huge_score is not None:
                    if gene in self.gene_to_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score is not None:
                    if gene in self.gene_to_gwas_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score_uncorrected is not None:
                    if gene in self.gene_to_gwas_huge_score_uncorrected:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score_uncorrected[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_exomes_huge_score is not None:
                    if gene in self.gene_to_exomes_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_exomes_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_positive_controls is not None:
                    if gene in self.gene_to_positive_controls:
                        line = "%s\t%.3g" % (line, self.gene_to_positive_controls[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.Y is not None:
                    line = "%s\t%s" % (line, "NA")
                if write_regression:
                    line = "%s\t%s" % (line, "NA")
                if self.Y_uncorrected is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.Y_w is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.Y_fw is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.priors_orig is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.priors_adj_orig is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.batches is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.X_orig is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.gene_to_chrom is not None:
                    line = "%s\t%s" % (line, self.gene_to_chrom[gene] if gene in self.gene_to_chrom else "NA")
                if self.gene_to_pos is not None:
                    line = "%s\t%s\t%s" % (line, self.gene_to_pos[gene][0] if gene in self.gene_to_pos else "NA", self.gene_to_pos[gene][1] if gene in self.gene_to_pos else "NA")
                    
                if self.gene_covariate_zs is not None:
                    line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(self.gene_covariate_names)) if j != self.gene_covariate_intercept_index]))

                output_fh.write("%s\n" % line)

    def write_gene_gene_set_statistics(self, output_file, max_no_write_gene_gene_set_beta=0.0001, write_filter_beta_uncorrected=False):
        log("Writing gene gene set stats to %s" % output_file, INFO)

        if self.genes is None or self.X_orig is None or (self.betas is None and self.beta_tildes is None):
            return

        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            gene_to_huge_score = self.gene_to_gwas_huge_score
            huge_score_label = "huge_score_gwas"
            gene_to_huge_score2 = self.gene_to_exomes_huge_score
            huge_score2_label = "huge_score_exomes"
        else:
            gene_to_huge_score = self.gene_to_huge_score
            huge_score_label = "huge_score"
            gene_to_huge_score2 = None
            huge_score2_label = None
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_gwas_huge_score
                huge_score_label = "huge_score_gwas"
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_exomes_huge_score
                huge_score_label = "huge_score_exomes"

        write_regression = self.Y_for_regression is not None and self.Y is not None and np.any(~np.isclose(self.Y, self.Y_for_regression))

        with open_gz(output_file, 'w') as output_fh:

            header = "Gene"

            if self.priors is not None:
                header = "%s\t%s" % (header, "prior")
            if self.combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "combined")
            if self.Y is not None:
                header = "%s\t%s" % (header, "log_bf")
            if write_regression:
                header = "%s\t%s" % (header, "log_bf_for_regression")
            if gene_to_huge_score is not None:
                header = "%s\t%s" % (header, huge_score_label)
            if gene_to_huge_score2 is not None:
                header = "%s\t%s" % (header, huge_score2_label)

            header = "%s\t%s\t%s\t%s" % (header, "gene_set", "beta", "weight")

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.genes))
            if self.combined_prior_Ys is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.combined_prior_Ys[k])
            elif self.priors is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.priors[k])
            elif self.Y is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y[k])
            elif write_regression is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y_for_regression[k])

            betas_to_use = self.betas if self.betas is not None else self.beta_tildes

            betas_for_filter = betas_to_use 
            if write_filter_beta_uncorrected and self.betas_uncorrected is not None:
                betas_for_filter = self.betas_uncorrected

            for i in ordered_i:
                gene = self.genes[i]

                if np.abs(self.X_orig[i,:]).sum() == 0:
                    continue

                ordered_j = sorted(self.X_orig[i,:].nonzero()[1], key=lambda k: -betas_to_use[k] / self.scale_factors[k])

                for j in ordered_j:
                    if np.abs(betas_for_filter[j] / self.scale_factors[j]) <= max_no_write_gene_gene_set_beta:
                        continue

                    line = gene
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, self.priors[i])
                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, self.combined_prior_Ys[i])
                    if self.Y is not None:
                        line = "%s\t%.3g" % (line, self.Y[i])
                    if write_regression:
                        line = "%s\t%.3g" % (line, self.Y_for_regression[i])
                    if gene_to_huge_score is not None:
                        huge_score = gene_to_huge_score[gene] if gene in gene_to_huge_score else 0
                        line = "%s\t%.3g" % (line, huge_score)
                    if gene_to_huge_score2 is not None:
                        huge_score2 = gene_to_huge_score2[gene] if gene in gene_to_huge_score2 else 0
                        line = "%s\t%.3g" % (line, huge_score2)


                    line = "%s\t%s\t%.3g\t%.3g" % (line, self.gene_sets[j], betas_to_use[j] / self.scale_factors[j], self.X_orig[i,j])
                    output_fh.write("%s\n" % line)

            ordered_i = range(len(self.genes_missing))
            if self.priors_missing is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.priors_missing[k])

            for i in ordered_i:
                gene = self.genes_missing[i]

                if np.abs(self.X_orig_missing_genes[i,:]).sum() == 0:
                    continue

                ordered_j = sorted(self.X_orig_missing_genes[i,:].nonzero()[1], key=lambda k: -betas_to_use[k] / self.scale_factors[k])

                for j in ordered_j:
                    if np.abs(betas_to_use[j] / self.scale_factors[j]) <= max_no_write_gene_gene_set_beta:
                        continue
                    line = gene
                    if self.priors is not None:
                        line = ("%s\t%.3g" % (line, self.priors_missing[i])) if self.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.combined_prior_Ys is not None:
                        line = ("%s\t%.3g" % (line, self.priors_missing[i])) if self.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.Y is not None:
                        line = "%s\t%s" % (line, "NA")
                    if write_regression:
                        line = "%s\t%s" % (line, "NA")
                    if gene_to_huge_score is not None:
                        line = "%s\t%s" % (line, "NA")
                    if gene_to_huge_score2 is not None:
                        line = "%s\t%s" % (line, "NA")

                    line = "%s\t%s\t%.3g\t%.3g" % (line, self.gene_sets[j], betas_to_use[j] / self.scale_factors[j], self.X_orig_missing_genes[i,j])
                    output_fh.write("%s\n" % line)


    def write_gene_set_overlap_statistics(self, output_file):
        log("Writing gene set overlap stats to %s" % output_file, INFO)
        with open_gz(output_file, 'w') as output_fh:
            if self.gene_sets is None:
                return
            if self.X_orig is None or self.betas is None or self.betas_uncorrected is None or self.mean_shifts is None or self.scale_factors is None:
                return
            header = "Gene_Set\tbeta\tbeta_uncorrected\tGene_Set_overlap\tV_beta\tV\tbeta_overlap\tbeta_uncorrected_overlap"
            output_fh.write("%s\n" % header)

            print_mask = self.betas_uncorrected != 0
            gene_sets = [self.gene_sets[i] for i in np.where(print_mask)[0]]
            X_to_print = self.X_orig[:,print_mask]
            mean_shifts = self.mean_shifts[print_mask]
            scale_factors = self.scale_factors[print_mask]
            betas_uncorrected = self.betas_uncorrected[print_mask]
            betas = self.betas[print_mask]

            num_batches = self._get_num_X_blocks(X_to_print)

            ordered_i = sorted(range(len(gene_sets)), key=lambda k: -betas[k] / scale_factors[k])

            gene_sets = [gene_sets[i] for i in ordered_i]
            X_to_print = X_to_print[:,ordered_i]
            mean_shifts = mean_shifts[ordered_i]
            scale_factors = scale_factors[ordered_i]
            betas_uncorrected = betas_uncorrected[ordered_i]
            betas = betas[ordered_i]

            for batch in range(num_batches):
                begin = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                if end > X_to_print.shape[1]:
                    end = X_to_print.shape[1]

                X_to_print[:,begin:end]
                mean_shifts[begin:end]
                scale_factors[begin:end]

                cur_V = self._compute_V(X_to_print[:,begin:end], mean_shifts[begin:end], scale_factors[begin:end], X_orig2=X_to_print, mean_shifts2=mean_shifts, scale_factors2=scale_factors)
                cur_V_beta = cur_V * betas

                for i in range(end - begin):
                    outer_ind = int(i + batch * self.batch_size)
                    ordered_j = sorted(np.where(cur_V_beta[i,:] != 0)[0], key=lambda k: -cur_V_beta[i,k] / scale_factors[k])
                    for j in ordered_j:
                        if outer_ind == j:
                            continue
                        output_fh.write("%s\t%.3g\t%.3g\t%s\t%.3g\t%.3g\t%.3g\t%.3g\n" % (gene_sets[outer_ind], betas[outer_ind] / scale_factors[outer_ind], betas_uncorrected[outer_ind] / scale_factors[outer_ind], gene_sets[j], cur_V_beta[i, j] / scale_factors[i], cur_V[i,j], betas[j] / scale_factors[j], betas_uncorrected[j] / scale_factors[j]))


    def write_gene_covariates(self, output_file):
        if self.genes is None or self.gene_covariates is None:
            return

        assert(self.gene_covariates.shape[1] == len(self.gene_covariate_names))
        log("Writing covs to %s" % output_file, INFO)

        with open_gz(output_file, 'w') as output_fh:

            #gene_covariate_betas = 
            #if self.gene_covariate_betas is not None:
            #    value_out = "#betas\tbetas"
            #    for j in range(self.gene_covariates.shape[1]):
            #        value_out += ("\t%.4g" % self.gene_covariate_betas[j])
            #    output_fh.write("%s\n" % value_out)

            header = "%s\t%s" % ("Gene\tin_regression", "\t".join(self.gene_covariate_names))
            output_fh.write("%s\n" % header)

            for i in range(len(self.genes)):
                value_out = "%s\t%s" % (self.genes[i], self.gene_covariates_mask[i])
                for j in range(self.gene_covariates.shape[1]):
                    value_out += ("\t%.4g" % self.gene_covariates[i,j])
                output_fh.write("%s\n" % value_out)


    def write_gene_effectors(self, output_file):
        if self.genes is None or self.huge_signal_bfs is None:
            return

        assert(self.huge_signal_bfs.shape[1] == len(self.huge_signals))
        
        log("Writing gene effectors to %s" % output_file, INFO)

        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            gene_to_huge_score = self.gene_to_gwas_huge_score
        else:
            gene_to_huge_score = self.gene_to_huge_score
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_gwas_huge_score
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_exomes_huge_score

        with open_gz(output_file, 'w') as output_fh:

            header = "Lead_locus\tInput\tP\tGene"

            if self.combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "cond_prob_total") #probability of each gene under assumption that only one is causal
            if self.Y is not None and self.priors is not None:
                header = "%s\t%s" % (header, "cond_prob_signal") #probability of each gene under assumption that only one is an effector (more than one could be causal if there are multiple SNPs each with different effectors)
            if self.priors is not None:
                header = "%s\t%s" % (header, "cond_prob_prior") #probability of each gene using only priors (assumption one causal gene)
            if gene_to_huge_score is not None:
                header = "%s\t%s" % (header, "cond_prob_huge") #probability of each gene using only distance/s2g (assumption one causal effector)
            if self.combined_Ds is not None:
                header = "%s\t%s" % (header, "combined_D")

            output_fh.write("%s\n" % header)

            for signal_ind in range(len(self.huge_signals)):

                gene_inds = self.huge_signal_bfs[:, signal_ind].nonzero()[0]

                max_log_bf = 10

                cond_prob_total = None
                if self.combined_prior_Ys is not None:
                    combined_prior_Y_bfs = self.combined_prior_Ys[gene_inds]
                    combined_prior_Y_bfs[combined_prior_Y_bfs > max_log_bf] = max_log_bf                    
                    combined_prior_Y_bfs = np.exp(combined_prior_Y_bfs)
                    cond_prob_total = combined_prior_Y_bfs / np.sum(combined_prior_Y_bfs)

                cond_prob_prior = None
                if self.priors is not None:
                    prior_bfs = self.priors[gene_inds]
                    prior_bfs[prior_bfs > max_log_bf] = max_log_bf                    
                    prior_bfs = np.exp(prior_bfs)
                    cond_prob_prior = prior_bfs / np.sum(prior_bfs)
                    
                    if self.Y is not None:
                        log_bf_bfs = self.huge_signal_bfs[:,signal_ind].todense().A1[gene_inds] * prior_bfs
                        cond_prob_log_bf = log_bf_bfs / np.sum(log_bf_bfs)

                cond_prob_huge = None
                if gene_to_huge_score is not None:
                    cond_prob_huge = self.huge_signal_bfs[:,signal_ind].todense().A1[gene_inds]
                    cond_prob_huge /= np.sum(cond_prob_huge)

                for i in range(len(gene_inds)):
                    gene_ind = gene_inds[i]
                    line = "%s:%d\t%s\t%.3g\t%s" % (self.huge_signals[signal_ind][0], self.huge_signals[signal_ind][1], self.huge_signals[signal_ind][3], self.huge_signals[signal_ind][2], self.genes[gene_ind])

                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, cond_prob_total[i])
                    if self.Y is not None and self.priors is not None:
                        line = "%s\t%.3g" % (line, cond_prob_log_bf[i])
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, cond_prob_prior[i])
                    if gene_to_huge_score is not None:
                        line = "%s\t%.3g" % (line, cond_prob_huge[i])
                    if self.combined_Ds is not None:
                        line = "%s\t%.3g" % (line, self.combined_Ds[gene_ind])

                    output_fh.write("%s\n" % line)


    def write_phewas_statistics(self, output_file):
        if self.phenos is None or len(self.phenos) == 0:
            return

        log("Writing phewas stats to %s" % output_file, INFO)

        with open_gz(output_file, 'w') as output_fh:

            header = "Pheno"

            ordered_inds = None

            write = False
            if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None:
                write = True
                ordered_inds = sorted(range(len(self.phenos)), key=lambda k: -self.pheno_Y_vs_input_combined_prior_Ys_beta[k])

            if self.pheno_Y_vs_input_Y_beta is not None:
                write = True
                ordered_inds = sorted(range(len(self.phenos)), key=lambda k: -self.pheno_Y_vs_input_Y_beta[k])

            if self.pheno_Y_vs_input_priors_beta is not None:
                write = True
                ordered_inds = sorted(range(len(self.phenos)), key=lambda k: -self.pheno_Y_vs_input_priors_beta[k])

            if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None:
                write = True
                ordered_inds = sorted(range(len(self.phenos)), key=lambda k: -self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta[k])

            if self.pheno_combined_prior_Ys_vs_input_Y_beta is not None:
                write = True
                ordered_inds = sorted(range(len(self.phenos)), key=lambda k: -self.pheno_combined_prior_Ys_vs_input_Y_beta[k])

            if self.pheno_combined_prior_Ys_vs_input_priors_beta is not None:
                write = True
                ordered_inds = sorted(range(len(self.phenos)), key=lambda k: -self.pheno_combined_prior_Ys_vs_input_priors_beta[k])

            if write:
                header = "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (header, "analysis", "beta_tilde", "P", "Z", "SE", "beta")


            if ordered_inds is None:
                ordered_inds = range(len(self.phenos))                                      

            output_fh.write("%s\n" % header)

            for i in ordered_inds:
                pheno = self.phenos[i]
                line = pheno
                if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "log_bf_vs_combined", self.pheno_Y_vs_input_combined_prior_Ys_beta_tilde[i], self.pheno_Y_vs_input_combined_prior_Ys_p_value[i], self.pheno_Y_vs_input_combined_prior_Ys_Z[i], self.pheno_Y_vs_input_combined_prior_Ys_se[i], self.pheno_Y_vs_input_combined_prior_Ys_beta[i]))

                if self.pheno_Y_vs_input_Y_beta is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "log_bf_vs_log_bf", self.pheno_Y_vs_input_Y_beta_tilde[i], self.pheno_Y_vs_input_Y_p_value[i], self.pheno_Y_vs_input_Y_Z[i], self.pheno_Y_vs_input_Y_se[i], self.pheno_Y_vs_input_Y_beta[i]))

                if self.pheno_Y_vs_input_priors_beta is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "log_bf_vs_prior", self.pheno_Y_vs_input_priors_beta_tilde[i], self.pheno_Y_vs_input_priors_p_value[i], self.pheno_Y_vs_input_priors_Z[i], self.pheno_Y_vs_input_priors_se[i], self.pheno_Y_vs_input_priors_beta[i]))

                if self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_vs_combined", self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde[i], self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value[i], self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z[i], self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se[i], self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta[i]))

                if self.pheno_combined_prior_Ys_vs_input_Y_beta is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_vs_log_bf", self.pheno_combined_prior_Ys_vs_input_Y_beta_tilde[i], self.pheno_combined_prior_Ys_vs_input_Y_p_value[i], self.pheno_combined_prior_Ys_vs_input_Y_Z[i], self.pheno_combined_prior_Ys_vs_input_Y_se[i], self.pheno_combined_prior_Ys_vs_input_Y_beta[i]))

                if self.pheno_combined_prior_Ys_vs_input_priors_beta is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_vs_prior", self.pheno_combined_prior_Ys_vs_input_priors_beta_tilde[i], self.pheno_combined_prior_Ys_vs_input_priors_p_value[i], self.pheno_combined_prior_Ys_vs_input_priors_Z[i], self.pheno_combined_prior_Ys_vs_input_priors_se[i], self.pheno_combined_prior_Ys_vs_input_priors_beta[i]))

    def write_factor_phewas_statistics(self, output_file):
        if self.phenos is None or len(self.phenos) == 0:
            return

        if self.factor_labels is None or (self.factor_phewas_Y_betas is None and self.factor_phewas_combined_prior_Ys_betas is None and self.factor_phewas_Y_huber_betas is None and self.factor_phewas_combined_prior_Ys_huber_betas is None):
            return 

        log("Writing factor phewas stats to %s" % output_file, INFO)

        with open_gz(output_file, 'w') as output_fh:

            header = "%s\t%s\t%s\t%s" % ("Factor", "Label", "Pheno", "analysis")

            header = "%s\t%s\t%s\t%s\t%s" % (header, "beta", "P", "Z", "SE")

            output_fh.write("%s\n" % header)

            for f in range(len(self.factor_labels)):
                if self.factor_phewas_Y_betas is not None:
                    ordered_fn = lambda k: self.factor_phewas_Y_p_values[f,k]
                else:
                    ordered_fn = lambda k: self.factor_phewas_combined_prior_Ys_p_values[f,k]

                for i in sorted(range(len(self.phenos)), key=ordered_fn):
                    pheno = self.phenos[i]
                    line = "%s\t%s\t%s" % ("Factor%d" % (f + 1), self.factor_labels[f], pheno)
                    if self.factor_phewas_Y_betas is not None:
                        output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "Y", self.factor_phewas_Y_betas[f,i], self.factor_phewas_Y_p_values[f,i], self.factor_phewas_Y_one_sided_p_values[f,i], self.factor_phewas_Y_zs[f,i], self.factor_phewas_Y_ses[f,i]))
                    if self.factor_phewas_Y_huber_betas is not None:
                        output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "Y_huber", self.factor_phewas_Y_huber_betas[f,i], self.factor_phewas_Y_huber_p_values[f,i], self.factor_phewas_Y_huber_one_sided_p_values[f,i], self.factor_phewas_Y_huber_zs[f,i], self.factor_phewas_Y_huber_ses[f,i]))
                    if self.factor_phewas_combined_prior_Ys_betas is not None:
                        output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined", self.factor_phewas_combined_prior_Ys_betas[f,i], self.factor_phewas_combined_prior_Ys_p_values[f,i], self.factor_phewas_combined_prior_Ys_one_sided_p_values[f,i], self.factor_phewas_combined_prior_Ys_zs[f,i], self.factor_phewas_combined_prior_Ys_ses[f,i]))
                    if self.factor_phewas_combined_prior_Ys_huber_betas is not None:
                        output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_huber", self.factor_phewas_combined_prior_Ys_huber_betas[f,i], self.factor_phewas_combined_prior_Ys_huber_p_values[f,i], self.factor_phewas_combined_prior_Ys_huber_one_sided_p_values[f,i], self.factor_phewas_combined_prior_Ys_huber_zs[f,i], self.factor_phewas_combined_prior_Ys_huber_ses[f,i]))


    def write_matrix_factors(self, factors_output_file=None, write_anchor_specific=False):

        if self.num_factors() <= 0:
            return

        anchors = None

        if write_anchor_specific:
            anchor_mask = self.anchor_pheno_mask if self.anchor_pheno_mask is not None else self.anchor_gene_mask

            if anchor_mask is None:
                anchors = ["default"]
            else:
                anchors = self.phenos if self.anchor_pheno_mask is not None else self.genes
                anchors = [anchors[x] for x in np.where(anchor_mask)[0]]

        ordered_inds = range(self.num_factors())

        if factors_output_file is not None:
            log("Writing factors to %s" % factors_output_file, INFO)
            with open_gz(factors_output_file, 'w') as output_fh:
                header = "Factor"
                header = "%s\t%s" % (header, "label")
                if anchors is not None:
                    header = "%s\t%s" % (header, "anchor")
                    header = "%s\t%s" % (header, "relevance")
                else:
                    header = "%s\t%s" % (header, "lambda")
                    header = "%s\t%s" % (header, "any_relevance")

                header = "%s\t%s" % (header, "top_genes")
                header = "%s\t%s" % (header, "top_gene_sets")
                output_fh.write("%s\n" % (header))
                    
                num_users = len(anchors) if anchors is not None else 1
                for j in range(num_users):
                    for i in ordered_inds:
                        line = "Factor%d" % (i+1)
                        line = "%s\t%s" % (line, self.factor_labels[i])
                        if anchors is not None:
                            line = "%s\t%s" % (line, anchors[j])
                            line = "%s\t%.3g" % (line, self.factor_anchor_relevance[i,j])
                            line = "%s\t%s" % (line, ",".join(self.factor_anchor_top_genes[i][j] if self.factor_anchor_top_genes is not None else self.factor_anchor_top_phenos[i][j]))
                            line = "%s\t%s" % (line, ",".join(self.factor_anchor_top_gene_sets[i][j]))

                        else:
                            line = "%s\t%.3g" % (line, self.exp_lambdak[i])
                            line = "%s\t%.3g" % (line, self.factor_relevance[i])
                            line = "%s\t%s" % (line, ",".join(self.factor_top_genes[i] if self.factor_top_genes is not None else self.factor_top_phenos[i]))
                            line = "%s\t%s" % (line, ",".join(self.factor_top_gene_sets[i]))
                        output_fh.write("%s\n" % (line))

    def write_clusters(self, gene_set_clusters_output_file=None, gene_clusters_output_file=None, pheno_clusters_output_file=None, write_anchor_specific=False, anchor_genes=None):

        if self.num_factors() == 0:
            log("No factors; not writing clusters")
            return

        anchors = None
        anchor_inds = None
        pheno_anchors = False
        gene_anchors = False

        if write_anchor_specific:
            anchor_mask = self.anchor_pheno_mask if self.anchor_pheno_mask is not None else self.anchor_gene_mask
            if anchor_mask is None:
                anchors = ["default"]
            else:
                if self.anchor_pheno_mask is not None:
                    anchors = self.phenos
                    pheno_anchors = True
                else:
                    anchors = self.genes
                    gene_anchors = True

                anchor_inds = np.where(anchor_mask)[0]
                anchors = [anchors[x] for x in anchor_inds]

        ordered_inds = range(self.num_factors())
        num_users = len(anchors) if anchors is not None else 1

        if gene_set_clusters_output_file is not None and self.exp_gene_set_factors is not None:
            
            #this uses value relative to others in the cluster
            #values_for_cluster = self.exp_gene_set_factors / np.sum(self.exp_gene_set_factors, axis=0)
            #this uses strongest absolute value
            values_for_cluster = self.exp_gene_set_factors

            log("Writing gene set clusters to %s" % gene_set_clusters_output_file, INFO)
            with open_gz(gene_set_clusters_output_file, 'w') as output_fh:
                gene_set_factor_gene_set_inds = list(range(self.exp_gene_set_factors.shape[0]))
                header = "Gene_Set"
                master_key_fn = None

                any_prob = None
                if anchors is None:
                    if self.betas is None and self.betas_uncorrected is None:
                        any_prob = 1 - np.prod(1 - self.gene_set_prob_factor_vector, axis=1)
                        header = "%s\t%s" % (header, "any_relevance")
                        master_key_fn = lambda k: -any_prob[k]

                if self.betas is not None or (pheno_anchors and self.X_phewas_beta is not None):
                    header = "%s\t%s" % (header, "beta")
                    if self.betas is not None:
                        master_key_fn = lambda k: -self.betas[gene_set_factor_gene_set_inds[k]]
                if self.betas_uncorrected is not None or (pheno_anchors and self.X_phewas_beta_uncorrected is not None):
                    header = "%s\t%s" % (header, "beta_uncorrected")
                    if self.betas_uncorrected is not None and master_key_fn is None:
                        master_key_fn = lambda k: -self.betas_uncorrected_[gene_set_factor_gene_set_inds[k]]

                if anchors is not None:
                    header = "%s\t%s" % (header, "relevance")

                header = "%s\t%s" % (header, "used_to_factor")

                if anchors is not None:
                    header = "%s\t%s" % (header, "anchor")

                output_fh.write("%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds])))

                if master_key_fn is None:
                    master_key_fn = lambda k: k

                for j in range(num_users):
                    if anchors is not None:
                        key_fn = lambda k: (-self.gene_set_prob_factor_vector[gene_set_factor_gene_set_inds[k],j], master_key_fn(k))
                    else:
                        key_fn = master_key_fn


                    for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):

                        #THINK WE CAN REMOVE THIS; DOES IT PASS ASSERT?
                        orig_i = gene_set_factor_gene_set_inds[i]
                        assert(orig_i == i)

                        line = self.gene_sets[orig_i]

                        if anchors is None:
                            if any_prob is not None:
                                line = "%s\t%.3g" % (line, any_prob[orig_i])
                            if self.betas is not None:
                                line = "%s\t%.3g" % (line, self.betas[orig_i])
                            if self.betas_uncorrected is not None:
                                line = "%s\t%.3g" % (line, self.betas_uncorrected[orig_i])

                        else:
                            if self.X_phewas_beta is not None and pheno_anchors:
                                line = "%s\t%.3g" % (line, self.X_phewas_beta[anchor_inds[j],orig_i])
                            elif self.betas is not None:
                                line = "%s\t%.3g" % (line, self.betas[orig_i])
                                
                            if self.X_phewas_beta_uncorrected is not None and pheno_anchors:
                                line = "%s\t%.3g" % (line, self.X_phewas_beta_uncorrected[anchor_inds[j],orig_i])
                            elif self.betas_uncorrected is not None:
                                line = "%s\t%.3g" % (line, self.betas_uncorrected[orig_i])

                            line = "%s\t%.3g" % (line, self.gene_set_prob_factor_vector[i,j])

                        used_to_factor = self.gene_set_factor_gene_set_mask[i] if self.gene_set_factor_gene_set_mask is not None else False
                        line = "%s\t%s" % (line, used_to_factor)

                        multiplier = 1
                        if anchors is not None:
                            line = "%s\t%s" % (line, anchors[j])
                            multiplier = self.gene_set_prob_factor_vector[orig_i,j]

                        cluster = np.argmax(values_for_cluster[i,:] * multiplier)

                        output_fh.write("%s\tFactor%d\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (multiplier * self.exp_gene_set_factors[i,k]) for k in ordered_inds])))

        if gene_clusters_output_file is not None and self.exp_gene_factors is not None:

            #this uses strongest absolute value
            values_for_cluster = self.exp_gene_factors

            log("Writing gene clusters to %s" % (gene_clusters_output_file), INFO)
            with open_gz(gene_clusters_output_file, 'w') as output_fh:
                gene_factor_gene_inds = list(range(self.exp_gene_factors.shape[0]))
                header = "Gene"
                master_key_fn = None

                any_prob = None
                if anchors is None:
                    if self.combined_prior_Ys is None and self.Y is None and self.priors is None:
                        any_prob = 1 - np.prod(1 - self.gene_prob_factor_vector, axis=1)
                        header = "%s\t%s" % (header, "any_relevance")
                        master_key_fn = lambda k: -any_prob[k]

                if self.combined_prior_Ys is not None or (pheno_anchors and self.gene_pheno_combined_prior_Ys is not None):
                    header = "%s\t%s" % (header, "combined")
                    if self.combined_prior_Ys is not None:
                        master_key_fn = lambda k: -self.combined_prior_Ys[gene_factor_gene_inds[k]]
                if self.Y is not None or (pheno_anchors and self.gene_pheno_Y is not None):
                    header = "%s\t%s" % (header, "log_bf")
                    if self.Y is not None and master_key_fn is None:
                        master_key_fn = lambda k: -self.Y[gene_factor_gene_inds[k]]
                if self.priors is not None or (pheno_anchors and self.gene_pheno_priors is not None):
                    header = "%s\t%s" % (header, "prior")
                    if self.priors is not None and master_key_fn is None:
                        master_key_fn = lambda k: -self.priors[gene_factor_gene_inds[k]]

                if anchors is not None:
                    header = "%s\t%s" % (header, "relevance")                    
                    header = "%s\t%s" % (header, "anchor")

                header = "%s\t%s" % (header, "used_to_factor")
                if gene_anchors:
                    header = "%s\t%s" % (header, "is_anchor")                    

                output_fh.write("%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds])))

                if master_key_fn is None:
                    master_key_fn = lambda k: k
                for j in range(num_users):
                    if anchors is not None:
                        key_fn = lambda k: (-self.gene_prob_factor_vector[gene_factor_gene_inds[k],j], master_key_fn(k))
                    else:
                        key_fn = master_key_fn

                    for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):

                        orig_i = gene_factor_gene_inds[i]
                        assert(orig_i == i)

                        line = self.genes[orig_i]

                        if anchors is None and any_prob is not None:
                            line = "%s\t%.3g" % (line, any_prob[orig_i])

                        if self.combined_prior_Ys is not None or (pheno_anchors and self.gene_pheno_combined_prior_Ys is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_combined_prior_Ys[orig_i,anchor_inds[j]] if pheno_anchors and self.gene_pheno_combined_prior_Ys is not None else self.combined_prior_Ys[orig_i])
                        if self.Y is not None or (pheno_anchors and self.gene_pheno_Y is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_Y[orig_i,anchor_inds[j]] if pheno_anchors and self.gene_pheno_Y is not None else self.Y[orig_i])
                        if self.priors is not None or (pheno_anchors and self.gene_pheno_priors is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_priors[orig_i,anchor_inds[j]] if pheno_anchors and self.gene_pheno_priors is not None else self.priors[orig_i])

                        multiplier = 1
                        if anchors is not None:
                            line = "%s\t%.3g" % (line, self.gene_prob_factor_vector[i,j])
                            line = "%s\t%s" % (line, anchors[j])
                            multiplier = self.gene_prob_factor_vector[orig_i,j]

                        used_to_factor = self.gene_factor_gene_mask[i] if self.gene_factor_gene_mask is not None else False
                        line = "%s\t%s" % (line, used_to_factor)

                        if gene_anchors:
                            line = "%s\t%s" % (line, anchor_mask[i])

                        cluster = np.argmax(values_for_cluster[i,:] * multiplier)
                        output_fh.write("%s\tFactor%d\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (multiplier * self.exp_gene_factors[i,k]) for k in ordered_inds])))

        if pheno_clusters_output_file is not None and self.exp_pheno_factors is not None:

            #this uses value relative to others in the cluster
            #this uses strongest absolute value
            values_for_cluster = self.exp_pheno_factors

            pheno_combined_prior_Ys = self.pheno_Y_vs_input_combined_prior_Ys_beta if self.pheno_Y_vs_input_combined_prior_Ys_beta is not None else self.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta
            pheno_Y = self.pheno_Y_vs_input_Y_beta if self.pheno_Y_vs_input_Y_beta is not None else self.pheno_combined_prior_Ys_vs_input_Y_beta
            pheno_priors = self.pheno_Y_vs_input_priors_beta if self.pheno_Y_vs_input_priors_beta is not None else self.pheno_combined_prior_Ys_vs_input_priors_beta            

            log("Writing pheno clusters to %s" % (pheno_clusters_output_file), INFO)
            with open_gz(pheno_clusters_output_file, 'w') as output_fh:
                pheno_factor_pheno_inds = list(range(self.exp_pheno_factors.shape[0]))
                header = "Pheno"
                master_key_fn = None

                any_prob = None

                if gene_anchors:
                    if self.gene_pheno_combined_prior_Ys is not None:
                        header = "%s\t%s" % (header, "combined")
                    if self.gene_pheno_Y is not None:
                        header = "%s\t%s" % (header, "log_bf")
                    if self.gene_pheno_priors is not None:
                        header = "%s\t%s" % (header, "prior")
                else:
                    if anchors is None and pheno_combined_prior_Ys is None and pheno_Y is None and pheno_priors is None:
                        any_prob = 1 - np.prod(1 - self.pheno_prob_factor_vector, axis=1)
                        header = "%s\t%s" % (header, "any_relevance")
                        master_key_fn = lambda k: -any_prob[k]

                    if pheno_combined_prior_Ys is not None:
                        header = "%s\t%s" % (header, "combined")
                        master_key_fn = lambda k: -pheno_combined_prior_Ys[pheno_factor_pheno_inds[k]]
                    if pheno_Y is not None:
                        header = "%s\t%s" % (header, "log_bf")
                        if master_key_fn is None:
                            master_key_fn = lambda k: -pheno_Y[pheno_factor_pheno_inds[k]]
                    if pheno_priors is not None:
                        header = "%s\t%s" % (header, "prior")
                        if master_key_fn is None:
                            master_key_fn = lambda k: -pheno_priors[pheno_factor_pheno_inds[k]]

                if anchors is not None:
                    header = "%s\t%s" % (header, "relevance")                    
                    header = "%s\t%s" % (header, "anchor")

                header = "%s\t%s" % (header, "used_to_factor")

                if pheno_anchors:
                    header = "%s\t%s" % (header, "is_anchor")

                output_fh.write("%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds])))

                if master_key_fn is None:
                    master_key_fn = lambda k: k
                for j in range(num_users):
                    if anchors is not None:
                        key_fn = lambda k: (-self.pheno_prob_factor_vector[pheno_factor_pheno_inds[k],j], master_key_fn(k))
                    else:
                        key_fn = master_key_fn
                    
                    for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):
                        #if np.sum(self.exp_pheno_factors[i,:]) == 0:
                        #    continue

                        orig_i = pheno_factor_pheno_inds[i]
                        assert(orig_i == i)

                        line = self.phenos[orig_i]

                        if not gene_anchors and anchors is None and any_prob is not None:
                            line = "%s\t%.3g" % (line, any_prob[orig_i])

                        if pheno_combined_prior_Ys is not None or (gene_anchors and self.gene_pheno_combined_prior_Ys is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_combined_prior_Ys[anchor_inds[j], orig_i] if gene_anchors and self.gene_pheno_combined_prior_Ys is not None else pheno_combined_prior_Ys[orig_i])
                        if pheno_Y is not None or (gene_anchors and self.gene_pheno_Y is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_Y[anchor_inds[j], orig_i] if gene_anchors and self.gene_pheno_Y is not None else pheno_Y[orig_i])
                        if pheno_priors is not None or (gene_anchors and self.gene_pheno_priors is not None):
                            line = "%s\t%.3g" % (line, self.gene_pheno_priors[anchor_inds[j], orig_i] if gene_anchors and self.gene_pheno_priors is not None else pheno_priors[orig_i])

                        multiplier = 1
                        if anchors is not None:
                            #relevance
                            line = "%s\t%.3g" % (line, self.pheno_prob_factor_vector[i,j])
                            line = "%s\t%s" % (line, anchors[j])
                            multiplier = self.pheno_prob_factor_vector[orig_i,j]

                        used_to_factor = self.pheno_factor_pheno_mask[i] if self.pheno_factor_pheno_mask is not None else False
                        line = "%s\t%s" % (line, used_to_factor)
                        if pheno_anchors:
                            line = "%s\t%s" % (line, anchor_mask[i])

                        cluster = np.argmax(values_for_cluster[i,:] * multiplier)
                        output_fh.write("%s\tFactor%d\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (multiplier * self.exp_pheno_factors[i,k]) for k in ordered_inds])))                    


    def write_gene_pheno_statistics(self, output_file=None, min_value_to_print=0):
        if self.gene_pheno_Y is None and self.gene_pheno_combined_prior_Ys is None and self.gene_pheno_priors is None:
            return

        if self.genes is None or self.phenos is None:
            return

        log("Writing gene pheno statistics to %s" % output_file)

        with open_gz(output_file, 'w') as output_fh:

            header = "Gene\tPheno"

            if self.gene_pheno_priors is not None:
                header = "%s\t%s" % (header, "prior")
            if self.gene_pheno_combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "combined")
            if self.gene_pheno_Y is not None:
                header = "%s\t%s" % (header, "log_bf")

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.genes))

            use_for_ordering = None

            if self.gene_pheno_combined_prior_Ys is not None:
                use_for_ordering = self.gene_pheno_combined_prior_Ys
            elif self.gene_pheno_priors is not None:
                use_for_ordering = self.gene_pheno_priors
            elif self.gene_pheno_Y is not None:
                use_for_ordering = self.gene_pheno_Y

            use_for_ordering_genes = use_for_ordering.max(axis=1).toarray().squeeze()

            ordered_i = sorted(ordered_i, key=lambda k: -np.max(use_for_ordering[k]))

            for i in ordered_i:
                gene = self.genes[i]
                ordered_j = range(len(self.phenos))
                ordered_j = sorted(ordered_j, key=lambda k: -use_for_ordering[i,k])
                for j in ordered_j:
                    pheno = self.phenos[j]
                    line = "%s\t%s" % (gene, pheno)
                    print_line = False
                    if self.gene_pheno_priors is not None:
                        line = "%s\t%.3g" % (line, self.gene_pheno_priors[i,j])
                        if self.gene_pheno_priors[i,j] > min_value_to_print:
                            print_line = True
                    if self.gene_pheno_combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, self.gene_pheno_combined_prior_Ys[i,j])
                        if self.gene_pheno_combined_prior_Ys[i,j] > min_value_to_print:
                            print_line = True
                    if self.gene_pheno_Y is not None:
                        line = "%s\t%.3g" % (line, self.gene_pheno_Y[i,j])
                        if self.gene_pheno_Y[i,j] > min_value_to_print:
                            print_line = True
                    if print_line:
                        output_fh.write("%s\n" % line)


    #HELPER FUNCTIONS

    '''
    Read in gene bfs for LOGISTIC or EMPIRICAL mapping
    '''
    def _record_param(self, param, value, overwrite=False, record_only_first_time=False):
        if param not in self.params:
            self.param_keys.append(param)
            self.params[param] = value
        elif record_only_first_time:
            return
        elif type(self.params[param]) == list:
            if self.params[param][-1] != value:
                self.params[param].append(value)
        elif self.params[param] != value:
            if overwrite:
                self.params[param] = value
            else:
                self.params[param] = [self.params[param], value]

    def _record_params(self, params, overwrite=False, record_only_first_time=False):
        for param in params:
            if params[param] is not None:
                self._record_param(param, params[param], overwrite=overwrite, record_only_first_time=record_only_first_time)

    def _read_gene_bfs(self, gene_bfs_in, gene_bfs_id_col=None, gene_bfs_log_bf_col=None, gene_bfs_combined_col=None, gene_bfs_prob_col=None, gene_bfs_prior_col=None, gene_bfs_sd_col=None, **kwargs):

        #require X matrix

        if gene_bfs_in is None:
            bail("Require --gene-bfs-in for this operation")

        log("Reading --gene-bfs-in file %s" % gene_bfs_in, INFO)
        gene_in_bfs = {}
        gene_in_combined = None
        gene_in_priors = None
        with open_gz(gene_bfs_in) as gene_bfs_fh:
            header_cols = gene_bfs_fh.readline().strip().split()
            if gene_bfs_id_col is None:
                gene_bfs_id_col = "Gene"

            id_col = self._get_col(gene_bfs_id_col, header_cols)

            prob_col = None
            if gene_bfs_prob_col is not None:
                prob_col = self._get_col(gene_bfs_prob_col, header_cols, True)

            bf_col = None
            if gene_bfs_log_bf_col is not None:
                bf_col = self._get_col(gene_bfs_log_bf_col, header_cols)
            else:
                if prob_col is None:
                    bf_col = self._get_col("log_bf", header_cols)

            if bf_col is None and prob_col is None:
                bail("--gene-bfs-bf-col or --gene-bfs-prob-col required for this operation")

            combined_col = None
            if gene_bfs_combined_col is not None:
                combined_col = self._get_col(gene_bfs_combined_col, header_cols, True)
            else:
                combined_col = self._get_col("combined", header_cols, False)

            prior_col = None
            if gene_bfs_prior_col is not None:
                prior_col = self._get_col(gene_bfs_prior_col, header_cols, True)
            else:
                prior_col = self._get_col("prior", header_cols, False)

            if combined_col is not None or prob_col is not None:
                gene_in_combined = {}
            if prior_col is not None:
                gene_in_priors = {}

            for line in gene_bfs_fh:
                cols = line.strip().split()
                if id_col >= len(cols) or (bf_col is not None and bf_col >= len(cols)) or (combined_col is not None and combined_col >= len(cols)) or (prob_col is not None and prob_col >= len(cols)) or (prior_col is not None and prior_col >= len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                if bf_col is not None:
                    try:
                        bf = float(cols[bf_col])
                    except ValueError:
                        if not cols[bf_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[bf_col], gene))
                        continue
                elif prob_col is not None:
                    try:
                        prob = float(cols[prob_col])
                    except ValueError:
                        if not cols[prob_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[prob_col], gene))
                        continue
                    if prob <= 0 or prob >= 1:
                        warn("Skipping probability %.3g outside of (0,1)" % (prob))
                        continue
                    bf = np.log(prob / (1 - prob)) - self.background_log_bf

                gene_in_bfs[gene] = bf

                if combined_col is not None:
                    try:
                        combined = float(cols[combined_col])
                    except ValueError:
                        if not cols[combined_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[combined_col], gene))
                        continue
                    gene_in_combined[gene] = combined

                if prior_col is not None:
                    try:
                        prior = float(cols[prior_col])
                    except ValueError:
                        if not cols[prior_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[prior_col], gene))
                        continue
                    gene_in_priors[gene] = prior


        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        gene_bfs = np.array([np.nan] * len(genes))
        
        extra_gene_bfs = []
        extra_genes = []
        for gene in gene_in_bfs:
            bf = gene_in_bfs[gene]
            if gene in gene_to_ind:
                gene_bfs[gene_to_ind[gene]] = bf
            else:
                extra_gene_bfs.append(bf)
                extra_genes.append(gene)

        return (gene_bfs, extra_genes, np.array(extra_gene_bfs), gene_in_combined, gene_in_priors)

    '''
    Read in gene Z scores for linear mapping
    '''
    def _read_gene_zs(self, gene_zs_in, gene_zs_id_col=None, gene_zs_value_col=None, background_95_prior=None, use_zs_as_log_odds=True, gws_threshold=None, gws_prob_true=None, max_mean_posterior=None, **kwargs):

        if gene_zs_in is None:
            bail("Require --gene-zs-in for this operation")

        log("Reading --gene-zs-in file %s" % gene_zs_in, INFO)
        if gws_threshold is not None and gws_prob_true is not None and max_mean_posterior is not None:
            use_zs_as_log_odds = False
            log("Mapping gene Z-scores to logistic probabilities", DEBUG)
        else:
            log("Treating gene Z-scores as raw log-odds", DEBUG)

        with open_gz(gene_zs_in) as gene_zs_fh:
            header_cols = gene_zs_fh.readline().strip().split()
            if gene_zs_id_col is None:
                bail("--gene-zs-id-col required for this operation")
            if gene_zs_value_col is None:
                bail("--gene-zs-value-col required for this operation")
            id_col = self._get_col(gene_zs_id_col, header_cols)
            value_col = self._get_col(gene_zs_value_col, header_cols)
            gene_zs = {}
            line_num_to_gene = []
            line_num = 0
            for line in gene_zs_fh:
                cols = line.strip().split()
                if id_col > len(cols) or value_col > len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                line_num_to_gene.append(gene)
                try:
                    value = float(cols[value_col])
                except ValueError:
                    if not cols[value_col] == "NA":
                        warn("Skipping unconvertible beta_tilde value %s for gene_set %s" % (cols[value_col], gene))
                    continue
                gene_zs[gene] = value
                line_num += 1

        if use_zs_as_log_odds:
            mean_z = np.mean(list(gene_zs.values()))
        else:
            for gene in gene_zs:
                gene_zs[gene] = np.abs(gene_zs[gene])
            #top posterior specifies what we want the posterior of the top ranked gene to actually be
            gws_z_threshold = -scipy.stats.norm.ppf(gws_threshold/2)
            median_z = np.median(np.array(list(gene_zs.values())))

            #logistic = L/(1 + exp(-k*(x-xo)))
            #Need:
            #L = asymptote (max allowed is 20%?)
            #then need two points to constrain it
            #average over all genes equals 0.05
            #gws threshold (how is this different from max?)
            #k = (log(1/y2 - L) - log(1/y1 - 1))/(x2-x1)
            #xo = (x1 * log(1/y2 - L) - x2 * log(1/y1 - L)) / (log(1/y2 - L) - log(1/y1 - L))

            x1 = median_z
            y1 = self.background_prior
            y1_bf = np.log(y1 / (1 - y1))

            x2 = gws_z_threshold
            y2 = gws_prob_true * max_mean_posterior
            y2_bf = np.log(y2 / (1 - y2))

            log("Fitting prop true logistic model with max bf=%.3g, points (%.3g,%.3g) and (%.3g,%.3g)" % (max_mean_posterior, x1, y1, x2, y2))

            L_param = max_mean_posterior
            k_param = (np.log(y2 / (L_param - y2)) - np.log(y1 / (L_param - y1))) / (x2 - x1)
            x_o_param = np.log((L_param - y2) / y2) / k_param + x2

            log("Using L=%.3g, k=%.3g, x_o=%.3g for logistic model of BF(Z)" % (L_param, k_param, x_o_param))

        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        gene_bfs = np.array([np.nan] * len(genes))
        extra_gene_bfs = []
        extra_genes = []
        #determine scale factor
        for gene in gene_zs:

            if use_zs_as_log_odds:
                bf = gene_zs[gene] - mean_z + self.background_log_bf
            else:
                posterior = L_param / (1 + np.exp(-k_param * (gene_zs[gene] - x_o_param)))
                bf = np.log(posterior / (1.0 - posterior)) - self.background_log_bf

            if gene in gene_to_ind:
                gene_bfs[gene_to_ind[gene]] = bf
            else:
                extra_gene_bfs.append(bf)
                extra_genes.append(gene)


        extra_gene_bfs = np.array(extra_gene_bfs)

        #if len(gene_bfs) == 0:
        #    bail("No genes in gene sets had percentiles!")

        return (gene_bfs, extra_genes, extra_gene_bfs)

    '''
    Read in gene percentiles for INVERSE_NORMALIZE mapping
    '''
    def _read_gene_percentiles(self, gene_percentiles_in, gene_percentiles_id_col=None, gene_percentiles_value_col=None, gene_percentiles_higher_is_better=False, top_posterior=0.99, min_prob=1e-4, max_prob=1-1e-4, **kwargs):

        if gene_percentiles_in is None:
            bail("Require --gene-percentiles-in for this operation")

        #top posterior specifies what we want the posterior of the top ranked gene to actually be
        top_log_pos_odd=np.log(top_posterior / (1-top_posterior))

        log("Reading --gene-percentiles-in file %s" % gene_percentiles_in, INFO)
        with open(gene_percentiles_in) as gene_percentiles_fh:
            header_cols = gene_percentiles_fh.readline().strip().split()
            if gene_percentiles_id_col is None:
                bail("--gene-percentiles-id-col required for this operation")
            if gene_percentiles_value_col is None:
                bail("--gene-percentiles-value-col required for this operation")
            id_col = self._get_col(gene_percentiles_id_col, header_cols)
            value_col = self._get_col(gene_percentiles_value_col, header_cols)
            gene_percentiles = {}
            line_num_to_gene = []
            line_num = 0
            for line in gene_percentiles_fh:
                cols = line.strip().split()
                if id_col > len(cols) or value_col > len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                line_num_to_gene.append(gene)
                try:
                    value = float(cols[value_col])
                except ValueError:
                    if not cols[value_col] == "NA":
                        warn("Skipping unconvertible beta_tilde value %s for gene_set %s" % (cols[value_col], gene))
                    continue
                gene_percentiles[gene] = value
                line_num += 1

            sorted_gene_percentiles = sorted(gene_percentiles.keys(), key=lambda x: gene_percentiles[x], reverse=gene_percentiles_higher_is_better)
            scale_factor=(top_log_pos_odd - self.background_log_bf)/scipy.stats.norm.ppf(float(len(sorted_gene_percentiles))/(len(sorted_gene_percentiles)+1))
            log("Using mean=%.3g, scale=%.3g for inverse normalized model of prob true" % (self.background_log_bf, scale_factor))

            #first gene is best
            for i in range(len(sorted_gene_percentiles)):
                gene_percentiles[sorted_gene_percentiles[i]] = 1 - float(i+1) / (len(sorted_gene_percentiles)+1)

            if self.genes is not None:
                genes = self.genes
                gene_to_ind = self.gene_to_ind
            else:
                genes = []
                gene_to_ind = {}

            gene_bf = np.array([np.nan] * len(genes))
            extra_gene_bf = []
            extra_genes = []
            #determine scale factor
            for gene in gene_percentiles:
                bf = scipy.stats.norm.ppf(gene_percentiles[gene], loc=self.background_log_bf, scale=scale_factor)
                if gene in gene_to_ind:
                    gene_bf[gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            extra_gene_bf = np.array(extra_gene_bf)

        if len(gene_bf) == 0:
            bail("No genes in gene sets had percentiles!")

        return (gene_bf, extra_genes, extra_gene_bf)

    def _read_gene_covs(self, gene_covs_in, gene_covs_id_col=None, gene_covs_cov_cols=None, **kwargs):

        #require X matrix

        if gene_covs_in is None:
            bail("Require --gene-covs-in for this operation")

        log("Reading --gene-covs-in file %s" % gene_covs_in, INFO)
        gene_in_covs = {}
        cov_names = []
        with open_gz(gene_covs_in) as gene_covs_fh:
            header_cols = gene_covs_fh.readline().strip().split()
            if gene_covs_id_col is None:
                gene_covs_id_col = "Gene"

            id_col = self._get_col(gene_covs_id_col, header_cols)

            cov_names = [header_cols[i] for i in range(len(header_cols)) if i != id_col]

            if len(cov_names) > 0:
                log("Read covariates %s" % (",".join(cov_names)), TRACE)

                for line in gene_covs_fh:
                    cols = line.strip().split()
                    if len(cols) != len(header_cols):
                        warn("Skipping due to too few columns in line: %s" % line)
                        continue

                    gene = cols[id_col]

                    covs = np.full(len(cov_names), np.nan)
                    try:
                        covs = np.array([float(cols[i]) for i in range(len(cols)) if i != id_col])
                    except ValueError:
                        continue

                    gene_in_covs[gene] = covs

        if len(cov_names) == 0:
            warn("No covariates in file")
            return

        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        gene_covs = np.full((len(genes), len(cov_names)), np.nan)
        
        extra_gene_covs = []
        extra_genes = []
        for gene in gene_in_covs:
            covs = gene_in_covs[gene]
            if gene in gene_to_ind:
                gene_covs[gene_to_ind[gene],:] = covs
            else:
                extra_gene_covs.append(covs)
                extra_genes.append(gene)

        return (cov_names, gene_covs, extra_genes, np.array(extra_gene_covs))


    def convert_prior_to_var(self, top_prior, num, frac):
        top_bf = np.log((top_prior) / (1 - top_prior)) - self.background_log_bf 
        if top_bf <= 0:
            bail("--top-gene-set-prior must be above background (%.4g)" % self.background_prior) 
        if frac is None:
            frac = 1
        if frac <= 0 or frac > 1:
            bail("--frac-gene-sets-for-prior must be in (0,1]")
        var = frac * np.square(top_bf / (-scipy.stats.norm.ppf(1.0 / (num * frac))))

        return var

    def _determine_columns(self, filename):
        #try to determine columns for gene_id, var_id, chrom, pos, p, beta, se, freq, n

        log("Trying to determine columns from headers and data for %s..." % filename)
        header = None
        with open_gz(filename) as fh:
            header = fh.readline().strip()
            orig_header_cols = header.split()

            first_line = fh.readline().strip()
            first_cols = first_line.split()

            if len(orig_header_cols) > len(first_cols):
                orig_header_cols = header.split('\t')

            header_cols = [x.strip('"').strip("'").strip() for x in orig_header_cols]
                
            def __get_possible_from_headers(header_cols, possible_headers1, possible_headers2=None):
                possible = np.full(len(header_cols), False)
                possible_inds = [i for i in range(len(header_cols)) if header_cols[i].lower().strip('_"') in possible_headers1]
                if len(possible_inds) == 0 and possible_headers2 is not None:
                    possible_inds = [i for i in range(len(header_cols)) if header_cols[i].lower() in possible_headers2]
                possible[possible_inds] = True
                return possible

            possible_gene_id_headers = set(["gene","id"])
            possible_var_id_headers = set(["var","id","rs", "varid"])
            possible_chrom_headers = set(["chr", "chrom", "chromosome", "#chrom"])
            possible_pos_headers = set(["pos", "bp", "position", "base_pair_location"])
            possible_locus_headers = set(["variant"])
            possible_p_headers = set(["p-val", "p_val", "pval", "p.value", "p-value", "p_value"])
            possible_p_headers2 = set(["p"])
            possible_beta_headers = set(["beta","effect"])
            possible_se_headers = set(["se","std", "stderr", "standard_error"])
            possible_freq_headers = set(["maf","freq"])
            possible_freq_headers2 = set(["af", "effect_allele_frequency"])
            possible_n_headers = set(["sample", "neff", "TotalSampleSize"])
            possible_n_headers2 = set(["n"])

            possible_gene_id_cols = __get_possible_from_headers(header_cols, possible_gene_id_headers)
            possible_var_id_cols = __get_possible_from_headers(header_cols, possible_var_id_headers)
            possible_chrom_cols = __get_possible_from_headers(header_cols, possible_chrom_headers)
            possible_locus_cols = __get_possible_from_headers(header_cols, possible_locus_headers)
            possible_pos_cols = __get_possible_from_headers(header_cols, possible_pos_headers)
            possible_p_cols = __get_possible_from_headers(header_cols, possible_p_headers, possible_p_headers2)
            possible_beta_cols = __get_possible_from_headers(header_cols, possible_beta_headers)
            possible_se_cols = __get_possible_from_headers(header_cols, possible_se_headers)
            possible_freq_cols = __get_possible_from_headers(header_cols, possible_freq_headers, possible_freq_headers2)
            possible_n_cols = __get_possible_from_headers(header_cols, possible_n_headers, possible_n_headers2)

            missing_vals = set(["", ".", "-", "na"])
            num_read = 0
            max_to_read = 1000

            for line in fh:
                cols = line.strip().split()
                seen_non_missing = False
                if len(cols) != len(header_cols):
                    cols = line.strip().split('\t')

                if len(cols) != len(header_cols):
                    bail("Error: couldn't parse line into same number of columns as header (%d vs. %d)" % (len(cols), len(header_cols)))

                for i in range(len(cols)):
                    token = cols[i].lower()

                    if token.lower() in missing_vals:
                        continue

                    seen_non_missing = True


                    if possible_gene_id_cols[i]:
                        try:
                            val = float(cols[i])
                            if not int(val) == val:
                                possible_gene_id_cols[i] = False
                        except ValueError:
                            pass
                    if possible_var_id_cols[i]:
                        if len(token) < 4:
                            possible_var_id_cols[i] = False

                        if "chr" in token or ":" in token or "rs" in token or "_" in token or "-" in token or "var" in token:
                            pass
                        else:
                            possible_var_id_cols[i] = False
                    if possible_chrom_cols[i]:
                        if "chr" in token or "x" in token or "y" in token or "m" in token:
                            pass
                        else:
                            try:
                                val = int(cols[i])
                                if val < 1 or val > 26:
                                    possible_chrom_cols[i] = False
                            except ValueError:
                                possible_chrom_cols[i] = False
                    if possible_locus_cols[i]:
                        if "chr" in token or "x" in token or "y" in token or "m" in token:
                            pass
                        else:
                            try:
                                locus = None
                                for delim in [":", "_"]:
                                    if delim in cols[i]:
                                        locus = cols[i].split(delim)
                                if locus is not None and len(locus) >= 2:
                                    chrom = int(locus[0])
                                    pos = int(locus[1])
                                    if chrom < 1 or chrom > 26:
                                        possible_locus_cols[i] = False
                            except ValueError:
                                possible_locus_cols[i] = False
                    if possible_pos_cols[i]:
                        try:
                            if len(token) < 3:
                                possible_pos_cols[i] = False
                            val = float(cols[i])
                            if not int(val) == val:
                                possible_pos_cols[i] = False
                        except ValueError:
                            possible_pos_cols[i] = False

                    if possible_p_cols[i]:
                        try:
                            val = float(cols[i])
                            if val > 1 or val < 0:
                                possible_p_cols[i] = False
                        except ValueError:
                            
                            possible_p_cols[i] = False
                    if possible_beta_cols[i]:
                        try:
                            val = float(cols[i])
                        except ValueError:
                            possible_beta_cols[i] = False
                    if possible_se_cols[i]:
                        try:
                            val = float(cols[i])
                            if val < 0:
                                possible_se_cols[i] = False
                        except ValueError:
                            possible_se_cols[i] = False
                    if possible_freq_cols[i]:
                        try:
                            val = float(cols[i])
                            if val > 1 or val < 0:
                                possible_freq_cols[i] = False
                        except ValueError:
                            possible_freq_cols[i] = False
                    if possible_n_cols[i]:
                        if len(token) < 3:
                            possible_n_cols[i] = False
                        else:
                            try:
                                val = float(cols[i])
                                if val < 0:
                                    possible_n_cols[i] = False
                            except ValueError:
                                possible_n_cols[i] = False
                if seen_non_missing:
                    num_read += 1
                    if num_read >= max_to_read:
                        break
                    
        possible_beta_cols[possible_p_cols] = False
        possible_beta_cols[possible_se_cols] = False
        possible_beta_cols[possible_pos_cols] = False

        total_possible = possible_gene_id_cols.astype(int) + possible_var_id_cols.astype(int) + possible_chrom_cols.astype(int) + possible_pos_cols.astype(int) + possible_p_cols.astype(int) + possible_beta_cols.astype(int) + possible_se_cols.astype(int) + possible_freq_cols.astype(int) + possible_n_cols.astype(int)
        for possible_cols in [possible_gene_id_cols, possible_var_id_cols, possible_chrom_cols, possible_pos_cols, possible_p_cols, possible_beta_cols, possible_se_cols, possible_freq_cols, possible_n_cols]:
            possible_cols[total_possible > 1] = False

        orig_header_cols = np.array(orig_header_cols)

        return (orig_header_cols[possible_gene_id_cols], orig_header_cols[possible_var_id_cols], orig_header_cols[possible_chrom_cols], orig_header_cols[possible_pos_cols], orig_header_cols[possible_locus_cols], orig_header_cols[possible_p_cols], orig_header_cols[possible_beta_cols], orig_header_cols[possible_se_cols], orig_header_cols[possible_freq_cols], orig_header_cols[possible_n_cols], header)

    def _adjust_bf(self, Y, min_mean_bf, max_mean_bf):
        Y_to_use = np.exp(Y)
        Y_mean = np.mean(Y_to_use)
        if min_mean_bf is not None and Y_mean < min_mean_bf:
            scale_factor = min_mean_bf / Y_mean
            log("Scaling up BFs by %.4g" % scale_factor)
            Y_to_use = Y_to_use * scale_factor
        elif max_mean_bf is not None and Y_mean > max_mean_bf:
            scale_factor = max_mean_bf / Y_mean
            log("Scaling down BFs by %.4g" % scale_factor)
            Y_to_use = Y_to_use * max_mean_bf / Y_mean
        return np.log(Y_to_use)

    def _complete_p_beta_se(self, p, beta, se):
        p_none_mask = np.logical_or(p == None, np.isnan(p))
        beta_none_mask = np.logical_or(beta == None, np.isnan(beta))
        se_none_mask = np.logical_or(se == None, np.isnan(se))

        se_zero_mask = np.logical_and(~se_none_mask, se == 0)
        se_zero_beta_non_zero_mask = np.logical_and(se_zero_mask, np.logical_and(~beta_none_mask, beta != 0))

        if np.sum(se_zero_beta_non_zero_mask) != 0:
            warn("%d variants had zero SEs; setting these to beta zero and se 1" % (np.sum(se_zero_beta_non_zero_mask)))
            beta[se_zero_beta_non_zero_mask] = 0
        se[se_zero_mask] = 1

        bad_mask = np.logical_and(np.logical_and(p_none_mask, beta_none_mask), se_none_mask)
        if np.sum(bad_mask) > 0:
            warn("Couldn't infer p/beta/se at %d positions; setting these to beta zero and se 1" % (np.sum(bad_mask)))
            p[bad_mask] = 1
            beta[bad_mask] = 0
            se[bad_mask] = 1
            p_none_mask[bad_mask] = False
            beta_none_mask[bad_mask] = False
            se_none_mask[bad_mask] = False

        if np.sum(p_none_mask) > 0:
            p[p_none_mask] = 2 * scipy.stats.norm.pdf(-np.abs(beta[p_none_mask] / se[p_none_mask]))
        if np.sum(beta_none_mask) > 0:
            z = np.abs(scipy.stats.norm.ppf(np.array(p[beta_none_mask]/2)))
            beta[beta_none_mask] = z * se[beta_none_mask]
        if np.sum(se_none_mask) > 0:
            z = np.abs(scipy.stats.norm.ppf(np.array(p[se_none_mask]/2)))
            z[z == 0] = 1
            se[se_none_mask] = np.abs(beta[se_none_mask] / z)
        return (p, beta, se)
        
    def _distill_huge_signal_bfs(self, huge_signal_bfs, huge_signal_posteriors, huge_signal_sum_gene_cond_probabilities, huge_signal_mean_gene_pos, huge_signal_max_closest_gene_prob, cap_region_posterior, scale_region_posterior, phantom_region_posterior, allow_evidence_of_absence, gene_covariates, gene_covariates_mask, gene_covariates_mat_inv, gene_covariate_names, gene_covariate_intercept_index, gene_prob_genes, total_genes=None, rel_prior_log_bf=None):

        if huge_signal_bfs is None:
            return

        if total_genes is not None:
            total_genes = self.genes

        #gene_to_ind = self._construct_map_to_ind(gene_prob_genes)


        if rel_prior_log_bf is None:
            prior_log_bf = np.full((1,huge_signal_bfs.shape[0]), self.background_log_bf)
        else:
            prior_log_bf = rel_prior_log_bf + self.background_log_bf
            if len(prior_log_bf.shape) == 1:
                prior_log_bf = prior_log_bf[np.newaxis,:]

        if prior_log_bf.shape[1] != huge_signal_bfs.shape[0]:
            bail("Error: priors shape did not match huge results shape (%s vs. %s)" % (prior_log_bf.shape, huge_signal_bfs.T.shape))

        if phantom_region_posterior:
            #first add an entry at the end to prior that is background_prior
            prior_log_bf = np.hstack((prior_log_bf, np.full((prior_log_bf.shape[0], 1), self.background_log_bf)))

            #then add a row to the bottom of signal_bfs
            phantom_probs = np.zeros(huge_signal_sum_gene_cond_probabilities.shape)
            phantom_mask = np.logical_and(huge_signal_sum_gene_cond_probabilities > 0, huge_signal_sum_gene_cond_probabilities < 1)

            phantom_probs[phantom_mask] = 1.0 - huge_signal_sum_gene_cond_probabilities[phantom_mask]

            #we need to set the BFs such that, when we add BFs below (with uniform prior) and then divide by total, we will get huge_signal_sum_gene_cond_probabilities for the non phantom
            #we *cannot* just convert phantom prob to phantom bf (like we do for signals; e.g. phantom_bfs = (phantom_probs / (1 - phantom_probs)) / self.background_bf) because the signals are defined as marginal probabilities
            #the BF needed to take gene in isolation from 0.05 to posterior
            #for phantom, we don't know the marginal -- it is inherently a joint estimate
            phantom_bfs = np.zeros(phantom_probs.shape)
            phantom_bfs[phantom_mask] = huge_signal_bfs.sum(axis=0).A1[phantom_mask] * (1.0 / huge_signal_sum_gene_cond_probabilities[phantom_mask] - 1.0)

            huge_signal_bfs = sparse.csc_matrix(sparse.vstack((huge_signal_bfs, phantom_bfs)))

            huge_signal_sum_gene_cond_probabilities = huge_signal_sum_gene_cond_probabilities + phantom_probs

        prior_bf = np.exp(prior_log_bf)

        prior = prior_bf / (1 + prior_bf)

        prior[prior == 1] = 1 - 1e-4
        prior[prior == 0] = 1e-4


        #utility sparse matrices to use within loop
        #huge results matrix has posteriors for the region
        signal_log_priors = sparse.csr_matrix(copy.copy(huge_signal_bfs).T)
        sparse_aux = copy.copy(signal_log_priors)

        huge_results = np.zeros(prior_log_bf.shape)
        for i in range(prior_log_bf.shape[0]):

            #need prior * (1 - other_prior)^N in each entry
            #due to sparse matrices, and limiting memory usage, have to overwrite 
            #also, the reason for the complication below is that we have to work in log space, which
            #requires addition rather than subtraction, which we can't do directly on sparse matrices
            #we also need to switch between operating on data (when we do pointwise operations)
            #and operating on matrices (when we sum across axes)

            #priors specific to the signal
            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(np.log(1 - prior[i,:])))
            other_log_priors = sparse_aux.sum(axis=1).A1

            signal_log_priors.data = np.ones(len(signal_log_priors.data))
            signal_log_priors = sparse.csr_matrix(signal_log_priors.multiply(np.log(prior[i,:])))

            #now this has log(prior/(1-prior))
            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            #now need to add in (1-prior)^N
            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.T.multiply(other_log_priors).T)

            #now this has log(prior * (1-other_prior)^N)
            signal_log_priors.data = signal_log_priors.data + sparse_aux.data

            #now normalize
            #log_sum_bf = c + np.log(np.sum(np.exp(var_log_bf[region_vars] - c)))
            #where c is max value

            #to get max, have to add minimum (to get it over zero) and then subtract it
            c = signal_log_priors.min(axis=1)

            #ensure all c are positive (otherwise this will be removed from the sparse matrix and break the subsequent operations on data)
            c = c.toarray()
            c[c == 0] = np.min(c) * 1e-4

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(c))

            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            c = signal_log_priors.max(axis=1) + c

            signal_log_priors.data = signal_log_priors.data + sparse_aux.data

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(c))
            #store the max
            c_data = copy.copy(sparse_aux.data)

            #subtract c
            sparse_aux.data = np.exp(signal_log_priors.data - c_data)

            norms = sparse_aux.sum(axis=1).A1
            norms[norms != 0] = np.log(norms[norms != 0])

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.T.multiply(norms).T)

            sparse_aux.data = c_data + sparse_aux.data

            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            #finally, we can obtain the priors matrix
            signal_log_priors.data = np.exp(signal_log_priors.data)            
            
            signal_priors = signal_log_priors.T

            #first have to adjust for priors
            #convert to BFs
            #we are overwriting data but drawing from the original (copied) huge_signal_bfs
            #multiply by priors. The final probabilities are proportional to the BFs * prior probabilities

            cur_huge_signal_bfs = huge_signal_bfs.multiply(signal_priors)

            #rescale; these are now posteriors for the signal
            #either:
            #1. sum to 1 (scale_region_posterior)
            #2. reduce (but don't increase) to 1 (cap_region_posterior)
            #3. leave them as is (but scale to be bayes factors before normalizing)

            new_norms = cur_huge_signal_bfs.sum(axis=0).A1

            if not scale_region_posterior and not cap_region_posterior:
                #treat them as bayes factors
                new_norms /= (huge_signal_mean_gene_pos * (np.mean(prior_bf[i:])/self.background_bf))

            #this scales everything to sum to 1

            #in case any have zero
            new_norms[new_norms == 0] = 1

            cur_huge_signal_bfs = sparse.csr_matrix(cur_huge_signal_bfs.multiply(1.0 / new_norms))

            if not scale_region_posterior and not cap_region_posterior:
                #convert them back to probabilities
                cur_huge_signal_bfs.data = cur_huge_signal_bfs.data / (1 + cur_huge_signal_bfs.data)

            #cur_huge_signal_bfs are actually now probabilities that sum to 1 (incorporating priors)
            #signal_cap_norm_factor incorporates both scaling to the signal prob, as well as any capping to reduce the probabilities to their original sum
            if cap_region_posterior:
                signal_cap_norm_factor = huge_signal_posteriors * huge_signal_sum_gene_cond_probabilities
            else:
                signal_cap_norm_factor = copy.copy(huge_signal_posteriors)

            #this is the "fudge factor" that accounts for the fact that the causal gene could be outside of this window
            #we don't need to do it under the phantom gene model because we already added a phantom gene to absorb 1 - max_closest_gene_prob

            max_per_signal = cur_huge_signal_bfs.max(axis=0).todense().A1 * signal_cap_norm_factor
            overflow_mask = max_per_signal > huge_signal_max_closest_gene_prob
            signal_cap_norm_factor[overflow_mask] *= (huge_signal_max_closest_gene_prob / max_per_signal[overflow_mask])

            #rescale to the signal probability (cur_huge_signal_posteriors is the probability that the signal is true)
            cur_huge_signal_bfs = sparse.csr_matrix(cur_huge_signal_bfs.multiply(signal_cap_norm_factor))


            if not allow_evidence_of_absence:
                #this part now ensures that nothing with absence of evidence has evidence of absence
                #consider two coin flips: first, it is causal due to the GWAS signal here
                #second, it is causal for some other reason
                #to not be causal, both need to come up negatuve
                #first has probability equal to huge_signal_posteriors
                #second has probability equal to prior

                #cur_huge_signal_bfs.data = 1 - (1 - np.array(cur_huge_signal_bfs.data)) * (1 - prior[i,:])
                #but, we cannot subtract from sparse matrices so have to do it this way

                cur_huge_signal_bfs.data = 1 - cur_huge_signal_bfs.data
                
                #cur_huge_signal_bfs = sparse.csc_matrix(cur_huge_signal_bfs.T.multiply(1 - prior[i,:]).T)
                cur_huge_signal_bfs = sparse.csc_matrix(cur_huge_signal_bfs.T.multiply(1 - self.background_prior).T)

                cur_huge_signal_bfs.data = 1 - cur_huge_signal_bfs.data            


            if cur_huge_signal_bfs.shape[1] > 0:
                #disable option to sum huge
                #if sum_huge:
                #    cur_huge_signal_bfs.data = np.log(1 - cur_huge_signal_bfs.data)
                #    huge_results[i,:] = 1 - np.exp(cur_huge_signal_bfs.sum(axis=1).A1)

                #This now has strongest signal posterior across all signals
                huge_results[i,:] = cur_huge_signal_bfs.max(axis=1).todense().A1


        #anything that was zero tacitly has probability equal to prior

        huge_results[huge_results == 0] = self.background_prior

        absent_genes = set()
        if total_genes is not None:
            #have to account for these
            absent_genes = set(total_genes) - set(gene_prob_genes)

        total_prob_causal = np.sum(huge_results)
        mean_prob_causal = (total_prob_causal + self.background_prior * len(absent_genes)) / (len(gene_prob_genes) + len(absent_genes)) 
        norm_constant = self.background_prior / mean_prob_causal

        #only normalize if enough genes
        max_prob = 1
        if len(gene_prob_genes) < 1000:
            norm_constant = max_prob
        elif norm_constant >= 1:
            norm_constant = max_prob

        #fix the maximum background prior across all genes
        #max_background_prior = None
        i#f max_background_prior is not None and mean_prob_causal > max_background_prior:
        #    norm_constant = max_background_prior / mean_prob_causal
        #else:

        norm_constant = max_prob

        if norm_constant != 1:
            log("Scaling output probabilities by %.4g" % norm_constant)

        huge_results *= norm_constant
        #now have to subtract out the prior

        okay_mask = huge_results < 1

        #we will add this to the prior to get the final posterior, so just subtract it
        huge_results[okay_mask] = np.log(huge_results[okay_mask] / (1 - huge_results[okay_mask])) - self.background_log_bf

        huge_results[~okay_mask] = np.max(huge_results[okay_mask])

        absent_prob = self.background_prior * norm_constant
        absent_log_bf = np.log(absent_prob / (1 - absent_prob)) - self.background_log_bf

        if phantom_region_posterior:
            huge_results = huge_results[:,:-1]

        huge_results_uncorrected = huge_results
        if gene_covariates is not None:
            (huge_results, huge_results_uncorrected, _) = self._correct_huge(huge_results, gene_covariates, gene_covariates_mask, gene_covariates_mat_inv, gene_covariate_names, gene_covariate_intercept_index)

        huge_results = np.squeeze(huge_results)
        huge_results_uncorrected = np.squeeze(huge_results_uncorrected)

        return (huge_results, huge_results_uncorrected, absent_genes, absent_log_bf)


    def _correct_huge(self, huge_results, gene_covariates, gene_covariates_mask, gene_covariates_mat_inv, gene_covariate_names, gene_covariate_intercept_index):

        if huge_results is None:
            return (None, None, None)

        if len(huge_results.shape) == 1:
            huge_results = huge_results[np.newaxis,:]

        huge_results_uncorrected = copy.copy(huge_results)
        gene_covariate_betas = None

        if gene_covariates is not None:
            assert(gene_covariates_mat_inv is not None)
            assert(gene_covariates_mask is not None)
            assert(gene_covariate_names is not None)
            assert(gene_covariate_intercept_index is not None)

            huge_results_mask = np.all(huge_results < np.mean(huge_results) + 5 * np.std(huge_results), axis=0)
            cur_gene_covariates_mask = np.logical_and(gene_covariates_mask, huge_results_mask)
            #dimensions are num_covariates x chains

            if self.huge_sparse_mode:
                pred_slopes = self.gene_covariate_slope_defaults.repeat(huge_results.shape[0]).reshape((len(self.gene_covariate_slope_defaults), huge_results.shape[0]))
            else:
                pred_slopes = gene_covariates_mat_inv.dot(gene_covariates[cur_gene_covariates_mask,:].T).dot(huge_results[:,cur_gene_covariates_mask].T)

            gene_covariate_betas = np.mean(pred_slopes, axis=1)
            log("Mean slopes are %s" % gene_covariate_betas, TRACE)

            non_intercept_inds = [i for i in range(len(gene_covariate_names)) if i != gene_covariate_intercept_index]

            param_names = ["%s_beta" % gene_covariate_names[i] for i in non_intercept_inds]
            param_values = gene_covariate_betas
            self._record_params(dict(zip(param_names, param_values)), record_only_first_time=True)

            pred_huge_adjusted = huge_results - gene_covariates[:,non_intercept_inds].dot(pred_slopes[non_intercept_inds,:]).T

            #flag those that are very high
            max_huge_change = 1.0

            bad_mask = pred_huge_adjusted - huge_results > max_huge_change

            if np.sum(bad_mask) > 0:
                warn("Not correcting %d genes for covariates due to large swings; there may be a problem with the covariates or input" % np.sum(bad_mask))

            huge_results[~bad_mask] = pred_huge_adjusted[~bad_mask]
            #JASON OLD
            #huge_results[~bad_mask] = pred_huge_residuals[~bad_mask]

        huge_results = np.squeeze(huge_results)
        huge_results_uncorrected = np.squeeze(huge_results_uncorrected)

        return (huge_results, huge_results_uncorrected, gene_covariate_betas)


    def _read_loc_file(self, loc_file, return_intervals=False, hold_out_chrom=None):

        gene_to_chrom = {}
        gene_to_pos = {}
        gene_chrom_name_pos = {}

        chrom_interval_to_gene = {}

        with open(loc_file) as loc_fh:
            for line in loc_fh:
                cols = line.strip().split()
                if len(cols) != 6:
                    bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                gene = cols[5]
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]
                chrom = self._clean_chrom(cols[1])
                if hold_out_chrom is not None and chrom == hold_out_chrom:
                    continue
                pos1 = int(cols[2])
                pos2 = int(cols[3])

                if gene in gene_to_chrom and gene_to_chrom[gene] != chrom:
                    warn("Gene %s appears multiple times with different chromosomes; keeping only first" % gene)
                    continue

                if gene in gene_to_pos and np.abs(np.mean(gene_to_pos[gene]) - np.mean((pos1,pos2))) > 1e7:
                    warn("Gene %s appears multiple times with far away positions; keeping only first" % gene)
                    continue

                gene_to_chrom[gene] = chrom
                gene_to_pos[gene] = (pos1, pos2)

                if chrom not in gene_chrom_name_pos:
                    gene_chrom_name_pos[chrom] = {}
                if gene not in gene_chrom_name_pos[chrom]:
                    gene_chrom_name_pos[chrom][gene] = set()
                if pos1 not in gene_chrom_name_pos[chrom][gene]:
                    gene_chrom_name_pos[chrom][gene].add(pos1)
                if pos2 not in gene_chrom_name_pos[chrom][gene]:
                    gene_chrom_name_pos[chrom][gene].add(pos2)

                if pos2 < pos1:
                    t = pos1
                    pos1 = pos2
                    pos2 = t

                if return_intervals:
                    if chrom not in chrom_interval_to_gene:
                        chrom_interval_to_gene[chrom] = {}

                    if (pos1, pos2) not in chrom_interval_to_gene[chrom]:
                        chrom_interval_to_gene[chrom][(pos1, pos2)] = []

                    chrom_interval_to_gene[chrom][(pos1, pos2)].append(gene) 

                #we consider distance to a gene to be both its start, its end, and also intermediate points within it
                #split_gene_length determines how many intermediate points there are
                split_gene_length = 1000000
                if pos2 > pos1:
                    for posm in range(pos1, pos2, split_gene_length)[1:]:
                        gene_chrom_name_pos[chrom][gene].add(posm)

        if return_intervals:
            return chrom_interval_to_gene
        else:
            return (gene_chrom_name_pos, gene_to_chrom, gene_to_pos)


    def _clean_chrom(self, chrom):
        if chrom[:3] == 'chr':
            return chrom[3:]
        else:
            return chrom

    def _read_correlations(self, gene_cor_file=None, gene_loc_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, compute_correlation_distance_function=True):
        if gene_cor_file is not None:
            log("Reading in correlations from %s" % gene_cor_file)
            unique_genes = np.array([True] * len(self.genes))
            correlation_m = [np.ones(len(self.genes))]
            with open(gene_cor_file) as gene_cor_fh:
                gene_cor_file_gene_col = gene_cor_file_gene_col - 1
                gene_cor_file_cor_start_col = gene_cor_file_cor_start_col - 1
                #store the genes in order, which we will need in order to map from each line in the file to correlation
                gene_cor_file_gene_names = []
                new_gene_index = {}
                cor_file_index = 0
                j = 0
                for line in gene_cor_fh:
                    if line[0] == "#":
                        continue
                    cols = line.strip().split()
                    if len(cols) < gene_cor_file_cor_start_col:
                        bail("Not enough columns in --gene-cor-file. Offending line:\n\t%s" % line)
                    gene_name = cols[gene_cor_file_gene_col]
                    if self.gene_label_map is not None and gene_name in self.gene_label_map:
                        gene_name = self.gene_label_map[gene_name]

                    gene_cor_file_gene_names.append(gene_name)
                    i = j - 1
                    if gene_name in self.gene_to_ind:
                        new_gene_index[gene_name] = cor_file_index
                        gene_correlations = [float(x) for x in cols[gene_cor_file_cor_start_col:]]
                        for gc_i in range(1,len(gene_correlations)+1):
                            cur_cor = gene_correlations[-gc_i]
                            if gc_i > cor_file_index:
                                bail("Error in --gene-cor-file: number of correlations is more than the number of genes seen to this point")
                            gene_i = gene_cor_file_gene_names[cor_file_index - gc_i]
                            if gene_i not in self.gene_to_ind:
                                continue
                            if cur_cor >= 1:
                                unique_genes[self.gene_to_ind[gene_i]] = False
                                #log("Excluding %s (correlation=%.4g with %s)" % (gene_i, cur_cor, gene_name), TRACE)

                            #store the values for the regression(s)
                            correlation_m_ind = j - i
                            while correlation_m_ind >= len(correlation_m):
                                correlation_m.append(np.zeros(len(self.genes)))
                            correlation_m[correlation_m_ind][i] = cur_cor
                            i -= 1
                        j += 1
                    cor_file_index += 1
            correlation_m = np.array(correlation_m)

            #now subset down the duplicate locations
            #self._subset_genes(unique_genes, skip_V=True, skip_scale_factors=True)
            #correlation_m = correlation_m[:,unique_genes]
            #log("Excluded %d duplicate genes" % sum(~unique_genes))

            sorted_gene_indices = sorted(range(len(self.genes)), key=lambda k: new_gene_index[self.genes[k]] if self.genes[k] in new_gene_index else 0)
            #sort the X and y values
            self._sort_genes(sorted_gene_indices, skip_V=True, skip_scale_factors=True)

        else:
            if gene_loc_file is None:
                bail("Need --gene-loc-file if don't specify --gene-cor-file")

            self.gene_locations = {}
            log("Reading gene locations")

            if self.gene_to_chrom is None:
                self.gene_to_chrom = {}
            if self.gene_to_pos is None:
                self.gene_to_pos = {}

            unique_genes = np.array([True] * len(self.genes))
            location_genes = {}
            with open(gene_loc_file) as gene_loc_fh:
                for line in gene_loc_fh:
                    cols = line.strip().split()
                    if len(cols) != 6:
                        bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                    gene_name = cols[5]
                    if gene_name not in self.gene_to_ind:
                        continue

                    chrom = cols[1]
                    start = int(cols[2])
                    end = int(cols[3])

                    self.gene_to_chrom[gene_name] = chrom
                    self.gene_to_pos[gene_name] = (start, end)

                    location = (chrom, start, end)
                    self.gene_locations[gene_name] = location
                    if location in location_genes:
                        #keep the one with highest Y
                        old_ind = self.gene_to_ind[location_genes[location]]
                        new_ind = self.gene_to_ind[gene_name]
                        if self.Y[old_ind] >= self.Y[new_ind]:
                            unique_genes[new_ind] = False
                            log("Excluding %s (duplicate of %s)" % (self.genes[new_ind], self.genes[old_ind]), TRACE)
                        else:
                            unique_genes[new_ind] = True
                            unique_genes[old_ind] = False
                            log("Excluding %s (duplicate of %s)" % (self.genes[old_ind], self.genes[new_ind]), TRACE)
                            location_genes[location] = gene_name
                    else:
                        location_genes[location] = gene_name

            #now subset down the duplicate locations
            self._subset_genes(unique_genes, skip_V=True, skip_scale_factors=True)

            sorted_gene_indices = sorted(range(len(self.genes)), key=lambda k: self.gene_locations[self.genes[k]] if self.genes[k] in self.gene_locations else ("NA", 0))

            #sort the X and y values
            self._sort_genes(sorted_gene_indices, skip_V=True, skip_scale_factors=True)

            #now we have to determine the relationship between distance and correlation
            correlation_m = self._compute_correlations_from_distance(compute_correlation_distance_function=compute_correlation_distance_function)

        #set the diagonal to 1
        correlation_m[0,:] = 1.0

        log("Banded correlation matrix: shape %s, %s" % (correlation_m.shape[0], correlation_m.shape[1]), DEBUG)
        log("Non-zero entries: %s" % sum(sum(correlation_m > 0)), DEBUG)

        return correlation_m

    def _compute_correlations_from_distance(self, Y=None, compute_correlation_distance_function=True):
        if self.genes is None:
            return None

        if Y is None:
            Y = self.Y

        if Y is None or self.gene_locations is None:
            return None

        if self.huge_sparse_mode:
            log("Too few genes from HuGE: using pre-computed correlation function", DEBUG)
            compute_correlation_distance_function = False

        correlation_m = [np.zeros(len(self.genes))]

        max_distance_to_model = 1000000.0
        num_bins = 1000
        distance_num = np.zeros(num_bins)
        distance_denom = np.zeros(num_bins)
        log("Calculating distance/correlation function")
        #this loop does two things
        #first, it stores the distances in a banded matrix -- this will be used later to compute the correlations
        #second, it stores the various distances / empirical covariances in two arrays for doing the regression
        for i in range(len(self.genes)):
            if self.genes[i] in self.gene_locations:
                loc = self.gene_locations[self.genes[i]]
                #traverse in each direction to find pairs within a range
                for j in range(i+1, len(self.genes)):
                    if self.genes[j] in self.gene_locations:
                        loc2 = self.gene_locations[self.genes[j]]
                        if not loc[0] == loc2[0]:
                            continue
                        distance = np.abs(loc2[1] - loc[1])
                        if distance > max_distance_to_model:
                            break
                        #store the values for the regression(s)
                        bin_number = int((distance / max_distance_to_model) * (num_bins - 1))
                        if Y[i] != 0:
                            distance_num[bin_number] += Y[i] * Y[j]
                            distance_denom[bin_number] += Y[i]**2
                        #store the distances for later
                        correlation_m_ind = j - i
                        while correlation_m_ind >= len(correlation_m):
                            correlation_m.append(np.array([np.inf] * len(self.genes)))
                        correlation_m[correlation_m_ind][i] = distance

        correlation_m = np.array(correlation_m)

        # fit function
        slope = -5.229e-07
        intercept = 0.54

        if compute_correlation_distance_function:
            bin_Y = distance_num[distance_denom != 0] / distance_denom[distance_denom != 0]
            bin_X = (np.array(range(len(distance_num))) * (max_distance_to_model / num_bins))[distance_denom != 0]
            sd_outlier_threshold = 3
            bin_outlier_max = np.mean(bin_Y) + sd_outlier_threshold * np.std(bin_Y)
            bin_mask = np.logical_and(bin_Y > -bin_outlier_max, bin_Y < bin_outlier_max)
            bin_Y = bin_Y[bin_mask]
            bin_X = bin_X[bin_mask]

            slope = np.cov(bin_X, bin_Y)[0,1] / np.var(bin_X)
            intercept = np.mean(bin_Y - bin_X * slope)
            max_distance = -intercept / slope
            if slope > 0:
                log("Slope was positive; setting all correlations to zero")
                intercept = 0
                slope = 0
            elif intercept < 0:
                log("Incercept was negative; setting all correlations to zero")                
                intercept = 0
                slope = 0
            else:
                log("Fit function from bins: r^2 = %.2g%.4gx; max distance=%d" % (intercept, slope, max_distance))
        else:
            max_distance = -intercept / slope
            log("Using precomputed function: r^2 = %.2g%.4gx; max distance=%d" % (intercept, slope, max_distance))

        if slope < 0:
            max_distance = -intercept / slope
            log("Using function: r^2 = %.2g + %.4g * x; max distance=%d" % (intercept, slope, max_distance))                
            self._record_params({"correlation_slope": slope, "correlation_intercept": intercept, "correlation_max_dist": max_distance})

            #map the values over from raw values to correlations/covariances
            correlation_m = intercept + slope * correlation_m
            correlation_m[correlation_m <= 0] = 0
            correlation_m[0,:] = 1.0
        else:
            correlation_m[0,:] = 1.0
            correlation_m[1:,:] = 0.0

        return correlation_m
            

    def _compute_beta_tildes(self, X, Y, y_var=None, scale_factors=None, mean_shifts=None, resid_correlation_matrix=None, log_fun=log):

        log_fun("Calculating beta tildes")

        if X.shape[0] == 0 or X.shape[1] == 0:
            bail("Can't compute beta tildes on no gene sets!")

        # Y can be a matrix with dimensions: 
        # number of parallel runs x number of genes
        #(k x n)
        #X is always a matrix with dimensions
        #(n x m)
        if len(Y.shape) == 2:
            len_Y = Y.shape[1]
            Y_mean = np.mean(Y, axis=1, keepdims=True)  # Compute row-wise mean
            #old_Y = (Y.T - np.mean(Y, axis=1)).T
        else:
            len_Y = Y.shape[0]
            Y_mean = np.mean(Y)
            #old_Y = Y - np.mean(Y)

        if mean_shifts is None or scale_factors is None:
            (mean_shifts, scale_factors) = self._calc_X_shift_scale(X)

        if y_var is None:
            if len(Y.shape) == 1:
                y_var = np.var(Y)
            else:
                y_var = np.var(Y, axis=1)

        # Update dot product to incorporate mean adjustment

        if sparse.issparse(X):
            X_sum = X.sum(axis=0).A1.T[:,np.newaxis]
        else:
            X_sum = np.asarray(X.sum(axis=0, keepdims=True).T)

        if len(Y.shape) == 1:
            X_sum = X_sum.squeeze(axis=1)

        dot_product = (X.T.dot(Y.T) - X_sum * Y_mean.T).T / len_Y

        variances = np.power(scale_factors, 2)

        #avoid divide by 0 only
        variances[variances == 0] = 1

        #multiply by scale factors because we store beta_tilde in units of scaled X
        beta_tildes = scale_factors * dot_product / variances

        if len(Y.shape) == 2:
            ses = np.outer(np.sqrt(y_var), scale_factors)
        else:
            ses = np.sqrt(y_var) * scale_factors

        ses /= (np.sqrt(variances * (len_Y - 1)))

        #FIXME: implement exact SEs
        #rather than just using y_var as a constant, calculate X.multiply(beta_tildes)
        #then, subtract out Y for non-zero entries, sum square, sum total
        #then, add in square of Y for zero entries, add in total
        #use these to calculate the variance term

        se_inflation_factors = None
        if resid_correlation_matrix is not None:
            log_fun("Adjusting standard errors for correlations", DEBUG)
            #need to multiply by inflation factors: (X * sigma * X) / variances

            #SEs and betas are stored in units of centered and scaled X
            #we do not need to scale X here, however, because cor_variances will then be in units of unscaled X
            #since variances are also in units of unscaled X, these will cancel out

            if type(resid_correlation_matrix) is list:
                resid_correlation_matrix_list = resid_correlation_matrix
                assert(len(resid_correlation_matrix_list) == beta_tildes.shape[0])
            else:
                resid_correlation_matrix_list = [resid_correlation_matrix]
                
            se_inflation_factors = np.zeros(beta_tildes.shape)

            for i in range(len(resid_correlation_matrix_list)):
                r_X = resid_correlation_matrix_list[i].dot(X)
                if sparse.issparse(X):
                    r_X_col_means = r_X.multiply(X).sum(axis=0).A1 / X.shape[0]
                else:
                    r_X_col_means = np.sum(r_X * X, axis=0) / X.shape[0]

                cor_variances = r_X_col_means - np.square(r_X_col_means)

                #never increase significance
                cor_variances[cor_variances < variances] = variances[cor_variances < variances]

                #both cor_variances and variances are in units of unscaled X
                cur_se_inflation_factors = np.sqrt(cor_variances / variances)

                if len(resid_correlation_matrix_list) == 1:
                    #these are all the same so just return
                    se_inflation_factors = cur_se_inflation_factors
                    if len(beta_tildes.shape) == 2:
                        se_inflation_factors = np.tile(se_inflation_factors, beta_tildes.shape[0]).reshape(beta_tildes.shape)
                    break
                else:
                    se_inflation_factors[i,:] = cur_se_inflation_factors

        return self._finalize_regression(beta_tildes, ses, se_inflation_factors)

    def _compute_multivariate_beta_tildes(self, X, Y, resid_correlation_matrix=None, add_intercept=True, covs=None):
        """
        Perform multivariate OLS regression of Y on X (plus optional covariates),
        optionally inflating standard errors via a sandwich formula using per-phenotype
        correlation matrices.

        Parameters
        ----------
        X : ndarray of shape (genes, factors)
            - Predictor matrix where rows are genes and columns are factors.

        Y : ndarray of shape (phenos, genes)
            - Outcome matrix where rows are phenotypes and columns are genes.

        resid_correlation_matrix : None or list of sparse/dense matrices
            - If provided, each entry is a (genes x genes) correlation matrix for a phenotype.
            - Used to inflate standard errors.

        covs : None or ndarray of shape (genes, n_covs)
            - Optional covariates where rows are covariates and columns are genes.

        Returns
        -------
        betas : ndarray of shape (phenos, k)
            - Regression coefficients for each phenotype and predictor.

        ses : ndarray of shape (phenos, k)
            - Standard errors for each coefficient.

        pvals : ndarray of shape (phenos, k)
            - Two-sided p-values for each coefficient.

        zscores : ndarray of shape (phenos, k)
            - Z-scores for each coefficient.
        """
        # ---------------------------
        # 1) Build the design matrix
        # ---------------------------
        # X is genes x factors
        # Transpose covariates to align with genes (genes x n_covs)
        if covs is not None:
            if len(covs.shape) == 1:
                covs = covs[:,np.newaxis]
            X_design = np.hstack([X, covs])  # shape (genes, factors + n_covs)
        else:
            X_design = X

        if add_intercept:
            ones_col = np.ones((X_design.shape[0], 1))  # shape (genes, 1)
            X_design = np.hstack([X_design, ones_col])  # (genes, factors + 1)


        n_obs, n_pred = X_design.shape  # genes, total predictors
        n_phenos = Y.shape[0]  # number of phenotypes (rows of Y)

        # --------------------------
        # 2) Compute OLS coefficients
        # --------------------------
        # Transpose Y to align with X (genes x phenos)
        Y_t = Y.T  # shape (genes, phenos)

        # Compute (X^T X) and its inverse
        XtX = X_design.T @ X_design  # shape (n_pred, n_pred)
        XtX_inv = np.linalg.inv(XtX)  # shape (n_pred, n_pred)

        # Compute (X^T Y)
        XtY = X_design.T @ Y_t  # shape (n_pred, phenos)

        # Compute beta coefficients
        betas = XtX_inv @ XtY  # shape (n_pred, phenos)
        betas = betas.T  # Transpose to (phenos, n_pred)

        # -----------------------------
        # 3) Compute residuals, SSE, df
        # -----------------------------
        # Residuals: Y_t - X_design @ betas.T (back to genes x phenos)
        fitted = X_design @ betas.T  # shape (genes, phenos)
        residuals = Y_t - fitted  # shape (genes, phenos)

        df = n_obs - n_pred  # degrees of freedom for classical OLS
        if df <= 0:
            raise ValueError("Degrees of freedom <= 0. Check the size of your input matrices.")

        # Sum of squared residuals (SSE) per phenotype
        sse = np.sum(residuals**2, axis=0)  # shape (phenos,)
        sigma2 = sse / df  # shape (phenos,)

        # -----------------------------
        # 4) Compute classical var(betas)
        # -----------------------------
        diag_xtx_inv = np.diag(XtX_inv)  # shape (n_pred,)

        # Classical standard errors
        classical_ses = np.sqrt(sigma2[:, None] * diag_xtx_inv[None, :])  # shape (phenos, n_pred)

        # By default, use classical standard errors
        final_ses = classical_ses.copy()

        # ---------------------------------------------------------------------
        # 5) If resid_correlation_matrix is provided, apply "sandwich" inflation
        # ---------------------------------------------------------------------
        if resid_correlation_matrix is not None:
            if len(resid_correlation_matrix) != n_phenos:
                raise ValueError(
                    "resid_correlation_matrix must be a list of length == n_phenos."
                )

            # Loop over phenotypes to apply the sandwich estimator
            for p in range(n_phenos):
                R_p = resid_correlation_matrix[p]  # shape (genes, genes)

                # Compute X^T (R_p X)
                if sparse.issparse(R_p):
                    XR_p = R_p.dot(X_design)  # shape (genes, n_pred)
                else:
                    XR_p = R_p @ X_design  # shape (genes, n_pred)

                XtR_pX = X_design.T @ XR_p  # shape (n_pred, n_pred)

                # Sandwich variance: (X^T X)^(-1) X^T R_p X (X^T X)^(-1)
                var_betas_p = XtX_inv @ XtR_pX @ XtX_inv  # shape (n_pred, n_pred)

                # Standard errors: sqrt of diagonal
                sandwich_se_p = np.sqrt(np.diag(var_betas_p))  # shape (n_pred,)

                # Update the final_ses for phenotype p
                final_ses[p, :] = sandwich_se_p

        # ------------------------------------
        # 6) Optionally strip out covariate betas
        # ------------------------------------
        if covs is not None or add_intercept:
            n_factors = X.shape[1]  # Number of factors (columns in X)
            betas = betas[:, :n_factors]  # Only the factor betas
            final_ses = final_ses[:, :n_factors]  # Corresponding standard errors

        # ------------------------------------
        # 7) Compute Z-scores and p-values
        # ------------------------------------
        #the inflation factors have already been accounted for above
        return self._finalize_regression(betas, final_ses, se_inflation_factors=None)

    def _compute_multivariate_beta_tildes_huber_correlated(self, X, Y, resid_correlation_matrix=None, covs=None, add_intercept=True, delta=1.0, max_iter=100, tol=1e-6, rel_tol=0.01):
        """
        Perform a naive "Huber + correlation" regression:
          1) Fit a Huber-type IRLS ignoring correlation.
          2) Post-hoc sandwich correction of standard errors using
             the provided sparse correlation matrices.

        Parameters
        ----------
        X : ndarray of shape (genes, factors)
            Rows represent genes, columns represent factors (predictors).

        Y : ndarray of shape (phenos, genes)
            Rows represent phenotypes, columns represent genes (observations).

        resid_correlation_matrix : list of sparse (genes x genes) or None
            If provided, must have length == number of phenotypes. Each
            entry is the correlation matrix for that phenotype. We apply
            a naive "sandwich" correction to the final standard errors.

        covs : ndarray of shape (genes, n_covs) or None
            Additional covariates where rows = covariates, cols = genes.

        add_intercept : bool
            If True, add a column of ones as intercept.

        delta : float
            Huber threshold parameter.

        max_iter : int
            Maximum IRLS iterations.

        tol : float
            Convergence tolerance (Frobenius norm difference in betas).

        Returns
        -------
        betas : ndarray of shape (phenos, k)
            Robust regression coefficients. (k = # factors + # covs + [1 if intercept])

        ses : ndarray of shape (phenos, k)
            "Sandwich"-adjusted standard errors (if resid_correlation_matrix is provided);
            otherwise, approximate robust SE ignoring correlation.

        pvals : ndarray of shape (phenos, k)
            Two-sided p-values for each coefficient.

        zscores : ndarray of shape (phenos, k)
            Z-scores for each coefficient.

        Notes
        -----
        - This method is *not* a rigorous robust + correlated approach. It simply:
           (a) obtains robust betas via Huber IRLS ignoring correlation,
           (b) applies a naive post-hoc "correlation sandwich" to the variance.

        - We carefully multiply by each R_p (which is sparse) to avoid blowing up memory.
        """

        # --------------------------------------------------------------------
        # 0) Build the design matrix: X_design
        # --------------------------------------------------------------------
        # X: (genes, factors)
        X_design = X
        if add_intercept:
            ones_col = np.ones((X.shape[0], 1))
            X_design = np.hstack([X_design, ones_col])  # now (genes, factors + 1)

        if covs is not None:
            # covs is (genes, n_covs) or (genes,) if you do covs[:, np.newaxis] above
            if len(covs.shape) == 1:
                covs = covs[:, np.newaxis]
            X_design = np.hstack([X_design, covs])  # final shape (genes, k)

        n_obs, n_pred = X_design.shape
        n_phenos = Y.shape[0]
        Y_t = Y.T  # (genes, phenos)

        # --------------------------------------------------------------------
        # 1) Huber IRLS ignoring correlation to get robust betas
        # --------------------------------------------------------------------
        def __huber_weight(resid, d):
            """w(r) = 1 if |r| <= d, else d / |r|."""
            w_ = np.ones_like(resid)
            mask_out = np.abs(resid) > d
            w_[mask_out] = d / np.abs(resid[mask_out])
            return w_

        def __huber_loss(resid, d):
            """
            piecewise huber:
              0.5*r^2 if |r| <= d,  d*(|r| - 0.5*d) otherwise.
            """
            out_ = np.zeros_like(resid)
            mask_in = np.abs(resid) <= d
            mask_out = ~mask_in
            out_[mask_in] = 0.5 * resid[mask_in]**2
            out_[mask_out] = d * (np.abs(resid[mask_out]) - 0.5*d)
            return out_

        # Initial guess: regular least squares
        # np.linalg.lstsq => returns coefs in shape (n_pred, phenos)
        W0, _, _, _ = np.linalg.lstsq(X_design, Y_t, rcond=None)
        betas_rob = W0.T  # => shape (phenos, n_pred)

        # X_expand shaped (phenos, genes, n_pred)
        # so axis=0 = phenos, axis=1 = genes, axis=2 = n_pred
        # We repeat along phenos dimension:
        X_expand = np.repeat(X_design[np.newaxis, :, :], n_phenos, axis=0)
        # => (phenos, genes, n_pred)

        for _ in range(max_iter):
            # shape => (genes, phenos)
            Y_hat = X_design @ betas_rob.T
            resid = Y_t - Y_hat

            # robust weights => shape (genes, phenos)
            w_ij_orig = __huber_weight(resid, delta)

            # Transpose to (phenos, genes) so it lines up with X_expand
            #   which is (phenos, genes, n_pred).
            w_ij = w_ij_orig.T  # => (phenos, genes)

            # Now broadcast: multiply each row j by w_ij[p, j]
            # X_expand is (phenos, genes, n_pred), w_ij is (phenos, genes)
            # => w_ij[..., None] => (phenos, genes, 1)
            X_expand_w = X_expand * w_ij[..., None]

            # X^T W X => shape (phenos, n_pred, n_pred)
            # We do "ijk, jh -> ikh"
            # i=phenos, j=genes, k=n_pred, h=n_pred
            XTwX = np.einsum('ijk,jh->ikh', X_expand_w, X_design)

            # X^T W Y => shape (phenos, n_pred)
            # "ijk, ji->ik" => i=phenos, j=genes, k=n_pred
            # But we want to multiply X_expand_w by Y_t => shape(genes, phenos)
            # => also note we need to use the shape from the same orientation
            XTwY = np.einsum('ijk,ji->ik', X_expand_w, Y_t)

            betas_new = np.zeros_like(betas_rob)  # (phenos, n_pred)
            for p in range(n_phenos):
                betas_new[p, :] = np.linalg.solve(XTwX[p], XTwY[p])

            diff = np.linalg.norm(betas_new - betas_rob, ord='fro')
            rel_diff = np.max(np.abs(W_new - W) / (np.abs(W_new) + np.abs(W) + 1e-20))

            betas_rob = betas_new
            log("Absolute diff=%.3g; rel_diff=%.3g" % (diff, rel_diff), TRACE)
            if diff < tol:
                break
            if rel_diff < rel_tol:
                break

        # Final residuals
        Y_hat = X_design @ betas_rob.T
        resid = Y_t - Y_hat

        # Huber "loss SSE"
        huber_vals = __huber_loss(resid, delta)  # (genes, phenos)
        sse = np.sum(huber_vals, axis=0)      # (phenos,)

        df = n_obs - n_pred
        if df <= 0:
            raise ValueError("Degrees of freedom <= 0; check input sizes.")

        # Approx. robust residual variance
        sigma2 = sse / df  # shape (phenos,)

        # We'll also need (X^T X)^{-1} for a quasi-variance approach
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)

        # 2) "Base" robust standard errors ignoring correlation
        diag_inv = np.diag(XtX_inv)  # (n_pred,)
        base_ses = np.sqrt(sigma2[:, None] * diag_inv[None, :])  # (phenos, n_pred)

        # 3) If no correlation matrix => Done
        #    Else do naive "sandwich" for correlated data
        if resid_correlation_matrix is None:
            final_ses = base_ses
        else:
            if len(resid_correlation_matrix) != n_phenos:
                raise ValueError("resid_correlation_matrix must match number of phenotypes.")

            final_ses = np.zeros_like(base_ses)  # shape (phenos, n_pred)

            # We'll reuse w_ij_orig for the final sandwich step, which is shape (genes, phenos)
            for p in range(n_phenos):
                R_p = resid_correlation_matrix[p]  # (genes, genes)
                # robust weights for phenotype p => w_ij_orig[:, p] => shape (genes,)
                w_vec = np.sqrt(w_ij_orig[:, p])

                # WeightedX => multiply each row i by w_vec[i]
                WeightedX = X_design * w_vec[:, None]  # shape (genes, n_pred)

                # Then multiply by R_p => shape => (genes, n_pred)
                if sparse.issparse(R_p):
                    WeightedX_R = R_p.dot(WeightedX)
                else:
                    WeightedX_R = R_p @ WeightedX

                # WeightedX^T * WeightedX_R => (n_pred, n_pred)
                XtRprimeX = WeightedX.T @ WeightedX_R

                var_betas_p = XtX_inv @ XtRprimeX @ XtX_inv
                se_p = np.sqrt(np.diag(var_betas_p))
                final_ses[p, :] = se_p

        # ------------------------------------
        # 6) Optionally strip out covariate betas
        # ------------------------------------
        if covs is not None or add_intercept:
            n_factors = X.shape[1]  # Number of factors (columns in X)
            betas_rob = betas_rob[:, :n_factors]  # Only the factor betas
            final_ses = final_ses[:, :n_factors]  # Corresponding standard errors

        # 4) Compute z-scores & p-values
        return self._finalize_regression(betas_rob, final_ses, se_inflation_factors=None)

    def _compute_robust_betas(self, X, Y, resid_correlation_matrix=None, covs=None, add_intercept=True, delta=1.0, max_iter=100, tol=1e-6, rel_tol=0.01):

        log("Calculating robust beta tildes", DEBUG)

        Y = Y.T
        if len(Y.shape) == 1:
            Y = Y[:,np.newaxis]

        n_phenos = Y.shape[1]

        #x is gene x factor
        n_factors = X.shape[1]  # Number of factors (columns in X)

        if add_intercept:
            X = np.hstack((X, np.ones((X.shape[0],1))))

        if covs is not None:
            if len(covs.shape) == 1:
                covs = covs[:,np.newaxis]
            X = np.hstack((X, covs))

        def _huber_loss(residuals, delta):
            return np.where(np.abs(residuals) <= delta, 0.5 * residuals ** 2, delta * (np.abs(residuals) - 0.5 * delta))

        def _huber_weight(residuals, delta):
            residuals[residuals == 0] = delta
            return np.where((np.abs(residuals) > 0) & (np.abs(residuals) <= delta), 1, delta / np.abs(residuals))

        # Initial coefficients
        W = np.linalg.lstsq(X, Y, rcond=None)[0]

        #pheno x gene x factor
        X_x_pheno = np.repeat(X[np.newaxis,:,:], Y.shape[1], axis=0)

        # Iteratively Reweighted Least Squares
        for iteration in range(max_iter):

            Y_pred = np.dot(X, W)
            residuals = Y - Y_pred
            weights = _huber_weight(residuals, delta)

            #unvectorized code for reference
            #W_new = np.zeros_like(W)
            #for i in range(Y.shape[1]):
            #    W_i = weights[:, i]
            #    XTWX = np.dot(X.T, np.multiply(X.T, W_i).T)
            #    XTWY = np.dot(X.T, np.multiply(W_i, Y[:, i]))
            #    W_new[:, i] = np.linalg.solve(XTWX, XTWY)


            #W is factor x phenos
            #weights is gene x phenos
            #Y is gene x phenos
            #X is gene x factor
            #X_x_pheno is pheno x gene x factor
            #X_x_pheno.T is factor x gene x pheno
            #weights are gene x factor

            #per pheno

            #pheno x gene x factor
            X_x_pheno_w = np.multiply(X_x_pheno.T, weights).T
            #X is factor x gene

            #pheno x factor x factor
            XTwX = np.einsum('pgf,gh->pfh', X_x_pheno_w, X)

            #gene x pheno 
            wY = np.multiply(weights, Y)
            #X_x_pheno is pheno x gene x factor

            XTwY = np.einsum('pgf,gp->fp', X_x_pheno, wY)

            #W_new = np.linalg.solve(XTwX, XTwY)

            #pheno x factor x factor
            XTwX_inv = np.linalg.inv(XTwX)
            W_new = np.einsum("phf,fp->hp", XTwX_inv, XTwY)

            if np.linalg.norm(W_new - W, ord='fro') < tol:
                break

            if np.max(np.abs(W_new - W) / (np.abs(W_new) + np.abs(W) + 1e-20)) < rel_tol:
                break


            W = W_new

        Y_pred = np.dot(X, W)
        residuals = Y - Y_pred
        betas = W.T

        # Calculate the variance of the residuals
        n = X.shape[0]
        p = X.shape[1]
        sse = np.sum(_huber_loss(residuals, delta), axis=0)
        #length equal to phenos
        sigma2 = sse / (n - p)


        # We'll also need (X^T X)^{-1} for a quasi-variance approach
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)

        # 2) "Base" robust standard errors ignoring correlation
        diag_inv = np.diag(XtX_inv)  # (n_pred,)
        base_ses = np.sqrt(sigma2[:, None] * diag_inv[None, :])  # (phenos, n_pred)

        if resid_correlation_matrix is None:
            ses = base_ses
        else:
            if len(resid_correlation_matrix) != n_phenos:
                raise ValueError("resid_correlation_matrix must match number of phenotypes.")

            ses = np.zeros_like(base_ses)  # shape (phenos, n_pred)

            # We'll reuse weights for the final sandwich step, which is shape (genes, phenos)
            for p in range(n_phenos):
                R_p = resid_correlation_matrix[p]  # (genes, genes)
                # robust weights for phenotype p => weights[:, p] => shape (genes,)
                w_vec = np.sqrt(weights[:, p])

                # WeightedX => multiply each row i by w_vec[i]
                WeightedX = X * w_vec[:, None]  # shape (genes, n_pred)

                # Then multiply by R_p => shape => (genes, n_pred)
                if sparse.issparse(R_p):
                    WeightedX_R = R_p.dot(WeightedX)
                else:
                    WeightedX_R = R_p @ WeightedX

                # WeightedX^T * WeightedX_R => (n_pred, n_pred)
                XtRprimeX = WeightedX.T @ WeightedX_R

                var_betas_p = XtX_inv @ XtRprimeX @ XtX_inv
                se_p = np.sqrt(np.diag(var_betas_p))
                ses[p, :] = se_p

        if covs is not None or add_intercept:
            betas = betas[:, :n_factors]  # Only the factor betas
            ses = ses[:, :n_factors]  # Corresponding standard errors

        return self._finalize_regression(betas, ses, se_inflation_factors=None)



    def _compute_logistic_beta_tildes(self, X, Y, scale_factors=None, mean_shifts=None, resid_correlation_matrix=None, convert_to_dichotomous=True, rel_tol=0.01, X_stacked=None, append_pseudo=True, log_fun=log):

        log_fun("Calculating logistic beta tildes")

        if X.shape[0] == 0 or X.shape[1] == 0:
            bail("Can't compute beta tildes on no gene sets!")

        if Y is self.Y or Y is self.Y_for_regression:
            Y = copy.copy(Y)

        if mean_shifts is None or scale_factors is None:
            (mean_shifts, scale_factors) = self._calc_X_shift_scale(X)            

        #Y can be a matrix with dimensions:
        #number of parallel runs x number of gene sets
        if len(Y.shape) == 1:
            orig_vector = True
            Y = Y[np.newaxis,:]
        else:
            orig_vector = False
        
        if convert_to_dichotomous:
            if np.sum(np.logical_and(Y != 0, Y != 1)) > 0:
                Y[np.isnan(Y)] = 0
                mult_sum = 1
                #log_fun("Multiplying Y sums by %.3g" % mult_sum)
                Y_sums = np.sum(Y, axis=1).astype(int) * mult_sum
                Y_sorted = np.sort(Y, axis=1)[:,::-1]
                Y_cumsum = np.cumsum(Y_sorted, axis=1)
                threshold_val = np.diag(Y_sorted[:,Y_sums])

                true_mask = (Y.T > threshold_val).T
                Y[true_mask] = 1
                Y[~true_mask] = 0
                log_fun("Converting values to dichotomous outcomes; y=1 for input y > %s" % threshold_val, DEBUG)

        log_fun("Outcomes: %d=1, %d=0; mean=%.3g" % (np.sum(Y==1), np.sum(Y==0), np.mean(Y)), TRACE)

        len_Y = Y.shape[1]
        num_chains = Y.shape[0]

        if append_pseudo:
            log_fun("Appending pseudo counts", TRACE)
            Y_means = np.mean(Y, axis=1)[:,np.newaxis]

            Y = np.hstack((Y, Y_means))

            #X = sparse.csc_matrix(sparse.vstack((X, np.ones(X.shape[1]))))
            X = sparse.csc_matrix(sparse.vstack((X, sparse.csr_matrix(np.ones((1, X.shape[1]))))))

            if X_stacked is not None:
                #X_stacked = sparse.csc_matrix(sparse.vstack((X_stacked, np.ones(X_stacked.shape[1]))))
                X_stacked = sparse.csc_matrix(sparse.vstack((X_stacked, sparse.csr_matrix(np.ones((1, X_stacked.shape[1]))))))

        #treat multiple chains as just additional gene set coefficients
        if X_stacked is None:
            if num_chains > 1:
                X_stacked = sparse.hstack([X] * num_chains)
            else:
                X_stacked = X

        num_non_zero = np.tile((X != 0).sum(axis=0).A1, num_chains)

        #old, memory more intensive
        #num_non_zero = (X_stacked != 0).sum(axis=0).A1

        num_zero = X_stacked.shape[0] - num_non_zero

        #initialize
        #one per gene set
        beta_tildes = np.zeros(X.shape[1] * num_chains)
        #one per gene set
        alpha_tildes = np.zeros(X.shape[1] * num_chains)
        it = 0

        compute_mask = np.full(len(beta_tildes), True)
        diverged_mask = np.full(len(beta_tildes), False)

        def __compute_Y_R(X, beta_tildes, alpha_tildes, max_cap=0.999):
            exp_X_stacked_beta_alpha = X.multiply(beta_tildes)
            exp_X_stacked_beta_alpha.data += (X != 0).multiply(alpha_tildes).data
            max_val = 100
            overflow_mask = exp_X_stacked_beta_alpha.data > max_val
            exp_X_stacked_beta_alpha.data[overflow_mask] = max_val
            np.exp(exp_X_stacked_beta_alpha.data, out=exp_X_stacked_beta_alpha.data)
            
            #each gene set corresponds to a 2 feature regression
            #Y/R_pred have dim (num_genes, num_chains * num_gene_sets)
            Y_pred = copy.copy(exp_X_stacked_beta_alpha)
            #add in intercepts
            Y_pred.data = Y_pred.data / (1 + Y_pred.data)
            Y_pred.data[Y_pred.data > max_cap] = max_cap
            R = copy.copy(Y_pred)
            R.data = Y_pred.data * (1 - Y_pred.data)
            return (Y_pred, R)

        def __compute_Y_R_zero(alpha_tildes):
            Y_pred_zero = np.exp(alpha_tildes)
            Y_pred_zero = Y_pred_zero / (1 + Y_pred_zero)
            R_zero = Y_pred_zero * (1 - Y_pred_zero)
            return (Y_pred_zero, R_zero)

        max_it = 100

        log_fun("Performing IRLS...")
        while True:
            it += 1
            prev_beta_tildes = copy.copy(beta_tildes)
            prev_alpha_tildes = copy.copy(alpha_tildes)

            #we are doing num_chains x X.shape[1] IRLS iterations in parallel.
            #Each parallel is a univariate regression of one gene set + intercept
            #first dimension is parallel chains
            #second dimension is each gene set as a univariate regression
            #calculate R
            #X is genesets*chains x genes

            #we are going to do this only for non-zero entries
            #the other entries are technically incorrect, but okay since we are only ever multiplying these by X (which have 0 at these entries)
            (Y_pred, R) = __compute_Y_R(X_stacked[:,compute_mask], beta_tildes[compute_mask], alpha_tildes[compute_mask])

            #values for the genes with zero for the gene set
            #these are constant across all genes (within a gene set/chain)
            max_val = 100
            overflow_mask = alpha_tildes > max_val
            alpha_tildes[overflow_mask] = max_val

            (Y_pred_zero, R_zero) = __compute_Y_R_zero(alpha_tildes[compute_mask])

            Y_sum_per_chain = np.sum(Y, axis=1)
            Y_sum = np.tile(Y_sum_per_chain, X.shape[1])

            #first term: phi*w in Bishop
            #This has length (num_chains * num_gene_sets)

            X_r_X_beta = X_stacked[:,compute_mask].power(2).multiply(R).sum(axis=0).A1.ravel()
            X_r_X_alpha = R.sum(axis=0).A1.ravel() + R_zero * num_zero[compute_mask]
            X_r_X_beta_alpha = X_stacked[:,compute_mask].multiply(R).sum(axis=0).A1.ravel()
            #inverse of [[a b] [c d]] is (1 / (ad - bc)) * [[d -b] [-c a]]
            #a = X_r_X_beta
            #b = c = X_r_X_beta_alpha
            #d = X_r_X_alpha
            denom = X_r_X_beta * X_r_X_alpha - np.square(X_r_X_beta_alpha)

            diverged = np.logical_or(np.logical_or(X_r_X_beta == 0, X_r_X_beta_alpha == 0), denom == 0)

            if np.sum(diverged) > 0:
                log_fun("%d beta_tildes diverged" % np.sum(diverged), TRACE)
                not_diverged = ~diverged

                cur_indices = np.where(compute_mask)[0]

                compute_mask[cur_indices[diverged]] = False
                diverged_mask[cur_indices[diverged]] = True

                #need to convert format in order to support indexing
                Y_pred = sparse.csc_matrix(Y_pred)
                R = sparse.csc_matrix(R)

                Y_pred = Y_pred[:,not_diverged]
                R = R[:,not_diverged]
                Y_pred_zero = Y_pred_zero[not_diverged]
                R_zero = R_zero[not_diverged]
                X_r_X_beta = X_r_X_beta[not_diverged]
                X_r_X_alpha = X_r_X_alpha[not_diverged]
                X_r_X_beta_alpha = X_r_X_beta_alpha[not_diverged]
                denom = denom[not_diverged]

            if np.sum(np.isnan(X_r_X_beta) | np.isnan(X_r_X_alpha) | np.isnan(X_r_X_beta_alpha)) > 0:
                bail("Error: something went wrong")

            #second term: r_inv * (y-t) in Bishop
            #for us, X.T.dot(Y_pred - Y)

            R_inv_Y_T_beta = X_stacked[:,compute_mask].multiply(Y_pred).sum(axis=0).A1.ravel() - X.T.dot(Y.T).T.ravel()[compute_mask]
            R_inv_Y_T_alpha = (Y_pred.sum(axis=0).A1.ravel() + Y_pred_zero * num_zero[compute_mask]) - Y_sum[compute_mask]

            beta_tilde_row = (X_r_X_beta * prev_beta_tildes[compute_mask] + X_r_X_beta_alpha * prev_alpha_tildes[compute_mask] - R_inv_Y_T_beta)
            alpha_tilde_row = (X_r_X_alpha * prev_alpha_tildes[compute_mask] + X_r_X_beta_alpha * prev_beta_tildes[compute_mask] - R_inv_Y_T_alpha)


            beta_tildes[compute_mask] = (X_r_X_alpha * beta_tilde_row - X_r_X_beta_alpha * alpha_tilde_row) / denom
            alpha_tildes[compute_mask] = (X_r_X_beta * alpha_tilde_row - X_r_X_beta_alpha * beta_tilde_row) / denom

            diff = np.abs(beta_tildes - prev_beta_tildes)
            diff_denom = np.abs(beta_tildes + prev_beta_tildes)
            diff_denom[diff_denom == 0] = 1
            rel_diff = diff / diff_denom

            #log_fun("%d left to compute; max diff=%.4g" % (np.sum(compute_mask), np.max(rel_diff)))
               
            compute_mask[np.logical_or(rel_diff < rel_tol, beta_tildes == 0)] = False
            if np.sum(compute_mask) == 0:
                log_fun("Converged after %d iterations" % it, TRACE)
                break
            if it == max_it:
                log_fun("Stopping with %d still not converged" % np.sum(compute_mask), TRACE)
                diverged_mask[compute_mask] = True
                break

        
        while True:
            #handle diverged
            if np.sum(diverged_mask) > 0:
                beta_tildes[diverged_mask] = 0
                alpha_tildes[diverged_mask] = Y_sum[diverged_mask] / len_Y

            max_coeff = 100            

            #genes x num_coeffs
            (Y_pred, V) = __compute_Y_R(X_stacked, beta_tildes, alpha_tildes)

            #this is supposed to calculate (X^t * V * X)-1
            #where X is the n x 2 matrix of genes x (1/0, 1)
            #d / (ad - bc) is inverse formula
            #a = X.multiply(V).multiply(X)
            #b = c = sum(X.multiply(V)
            #d = V.sum() + constant values for all zero X (since those aren't in V)
            #also need to add in enough p*(1-p) values for all of the X=0 entries; this is where the p_const * number of zero X comes in

            params_too_large_mask = np.logical_or(np.abs(alpha_tildes) > max_coeff, np.abs(beta_tildes) > max_coeff)
            #to prevent overflow
            alpha_tildes[np.abs(alpha_tildes) > max_coeff] = max_coeff

            p_const = np.exp(alpha_tildes) / (1 + np.exp(alpha_tildes))

            variance_denom = (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))
            denom_zero = variance_denom == 0
            variance_denom[denom_zero] = 1

            variances = X_stacked.power(2).multiply(V).sum(axis=0).A1 - np.power(X_stacked.multiply(V).sum(axis=0).A1, 2) / variance_denom
            variances[denom_zero] = 100

            #set them to diverged also if variances are negative or variance denom is 0 or if params are too large
            additional_diverged_mask = np.logical_and(~diverged_mask, np.logical_or(np.logical_or(variances < 0, denom_zero), params_too_large_mask))

            #get the likelihoods for the LRT
            #null_alpha = np.mean(Y, axis=1)
            #(Y_null_pred, V_null) = __compute_Y_R(X_stacked, 0, null_alpha)
            #null_likelihood = Y 
            #bail("")

            if np.sum(additional_diverged_mask) > 0:
                #additional divergences
                diverged_mask = np.logical_or(diverged_mask, additional_diverged_mask)
            else:
                break

        se_inflation_factors = None
        if resid_correlation_matrix is not None:
            if type(resid_correlation_matrix) is list:
                raise NotImplementedError("Vectorized correlations not yet implemented for logistic regression")

            log("Adjusting standard errors for correlations", DEBUG)
            #need to multiply by inflation factors: (X * sigma * X) / variances

            if append_pseudo:
                resid_correlation_matrix = sparse.hstack((resid_correlation_matrix, np.zeros(resid_correlation_matrix.shape[0])[:,np.newaxis]))
                new_bottom_row = np.zeros((1, resid_correlation_matrix.shape[1]))

                new_bottom_row[-1] = 1
                resid_correlation_matrix = sparse.vstack((resid_correlation_matrix, new_bottom_row)).tocsc()

            cor_variances = copy.copy(variances)

            r_X = resid_correlation_matrix.dot(X)
            #we will only be using this to multiply matrices that are non-zero only when X is
            r_X = (X != 0).multiply(r_X)

            cor_variances = sparse.hstack([r_X.multiply(X)] * num_chains).multiply(V).sum(axis=0).A1 - sparse.hstack([r_X] * num_chains).multiply(V).sum(axis=0).A1 / (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))

            #both cor_variances and variances are in units of unscaled X
            variances[variances == 0] = 1
            se_inflation_factors = np.sqrt(cor_variances / variances)

        
        #now unpack the chains

        if num_chains > 1:
            beta_tildes = beta_tildes.reshape(num_chains, X.shape[1])
            alpha_tildes = alpha_tildes.reshape(num_chains, X.shape[1])
            variances = variances.reshape(num_chains, X.shape[1])
            diverged_mask = diverged_mask.reshape(num_chains, X.shape[1])
            if se_inflation_factors is not None:
                se_inflation_factors = se_inflation_factors.reshape(num_chains, X.shape[1])
        else:
            beta_tildes = beta_tildes[np.newaxis,:]
            alpha_tildes = alpha_tildes[np.newaxis,:]
            variances = variances[np.newaxis,:]
            diverged_mask = diverged_mask[np.newaxis,:]
            if se_inflation_factors is not None:
                se_inflation_factors = se_inflation_factors[np.newaxis,:]

        variances[:,scale_factors == 0] = 1

        #not necessary
        #if inflate_se:
        #    inflate_mask = scale_factors > np.mean(scale_factors)
        #    variances[:,inflate_mask] *= np.mean(np.power(scale_factors, 2)) / np.power(scale_factors[inflate_mask], 2)  

        #multiply by scale factors because we store beta_tilde in units of scaled X
        beta_tildes = scale_factors * beta_tildes

        ses = scale_factors / np.sqrt(variances) 

        if orig_vector:
            beta_tildes = np.squeeze(beta_tildes, axis=0)
            alpha_tildes = np.squeeze(alpha_tildes, axis=0)
            variances = np.squeeze(variances, axis=0)
            ses = np.squeeze(ses, axis=0)
            diverged_mask = np.squeeze(diverged_mask, axis=0)

            if se_inflation_factors is not None:
                se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

        return self._finalize_regression(beta_tildes, ses, se_inflation_factors) + (alpha_tildes, diverged_mask)


    def _finalize_regression(self, beta_tildes, ses, se_inflation_factors):

        if se_inflation_factors is not None:
            ses *= se_inflation_factors

        if np.prod(ses.shape) > 0:
            #empty mask
            empty_mask = np.logical_and(beta_tildes == 0, ses <= 0)
            max_se = np.max(ses)

            if np.sum(empty_mask) > 0:
                log("Zeroing out %d betas due to negative ses" % (np.sum(empty_mask)), TRACE)

            ses[empty_mask] = max_se * 100 if max_se > 0 else 100

            #if no y var, set beta tilde to 0

            beta_tildes[ses <= 0] = 0

        z_scores = np.zeros(beta_tildes.shape)
        ses_positive_mask = ses > 0
        z_scores[ses_positive_mask] = beta_tildes[ses_positive_mask] / ses[ses_positive_mask]
        if np.any(~ses_positive_mask):
            warn("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))
        p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))
        return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)

    def _correct_beta_tildes(self, beta_tildes, ses, se_inflation_factors, total_qc_metrics, total_qc_metrics_directions, correct_mean=True, correct_var=True, add_missing=True, add_ignored=True, correct_ignored=False, fit=True):

        orig_total_qc_metrics = total_qc_metrics

        #inputs have first axis equal to number of chains
        if len(beta_tildes.shape) == 1:
            beta_tildes = beta_tildes[np.newaxis,:]
        if len(ses.shape) == 1:
            ses = ses[np.newaxis,:]
        if se_inflation_factors is not None and len(se_inflation_factors.shape) == 1:
            se_inflation_factors = se_inflation_factors[np.newaxis,:]

        remove_mask = np.full(beta_tildes.shape[1], False)

        if total_qc_metrics is None:
            if self.gene_covariates is None:
                warn("--correct-huge was not used, so skipping correction")
        else:
            if fit or self.total_qc_metric_betas is None:
                if add_missing:
                    if self.beta_tildes_missing is not None:
                        beta_tildes = np.hstack((beta_tildes, np.tile(self.beta_tildes_missing, beta_tildes.shape[0]).reshape(beta_tildes.shape[0], len(self.beta_tildes_missing))))
                        ses = np.hstack((ses, np.tile(self.ses_missing, ses.shape[0]).reshape(ses.shape[0], len(self.ses_missing))))
                        if se_inflation_factors is not None:
                            se_inflation_factors = np.hstack((se_inflation_factors, np.tile(self.se_inflation_factors_missing, se_inflation_factors.shape[0]).reshape(se_inflation_factors.shape[0], len(self.se_inflation_factors_missing))))

                        total_qc_metrics = np.vstack((total_qc_metrics, self.total_qc_metrics_missing))
                        remove_mask = np.append(remove_mask, np.full(len(self.beta_tildes_missing), True))

                if add_ignored:
                    if self.beta_tildes_ignored is not None:
                        beta_tildes = np.hstack((beta_tildes, np.tile(self.beta_tildes_ignored, beta_tildes.shape[0]).reshape(beta_tildes.shape[0], len(self.beta_tildes_ignored))))
                        ses = np.hstack((ses, np.tile(self.ses_ignored, ses.shape[0]).reshape(ses.shape[0], len(self.ses_ignored))))
                        if se_inflation_factors is not None:
                            se_inflation_factors = np.hstack((se_inflation_factors, np.tile(self.se_inflation_factors_ignored, se_inflation_factors.shape[0]).reshape(se_inflation_factors.shape[0], len(self.se_inflation_factors_ignored))))

                        total_qc_metrics = np.vstack((total_qc_metrics, self.total_qc_metrics_ignored))
                        remove_mask = np.append(remove_mask, np.full(len(self.beta_tildes_ignored), True))

                z_scores = np.zeros(beta_tildes.shape)
                z_scores[ses != 0] = np.abs(beta_tildes[ses != 0]) / ses[ses != 0]

                if self.huge_sparse_mode:
                    log("Too few genes from HuGE: using pre-computed correct betas", DEBUG)
                    self.total_qc_metric_intercept = self.total_qc_metric_intercept_defaults
                    self.total_qc_metric2_intercept = self.total_qc_metric2_intercept_defaults
                    self.total_qc_metric_betas = self.total_qc_metric_betas_defaults
                    self.total_qc_metric2_betas = self.total_qc_metric2_betas_defaults
                else:
                    z_scores_mask = np.all(np.logical_and(np.abs(z_scores - np.mean(z_scores)) <= 5 * np.std(z_scores), ses != 0), axis=0)
                    metrics_mask = np.all(np.abs(total_qc_metrics - np.mean(total_qc_metrics, axis=0)) <= 5 * np.std(total_qc_metrics, axis=0), axis=1)
                    pred_mask = np.logical_and(z_scores_mask, metrics_mask)

                    #find the intercept index
                    intercept_mask = (np.std(total_qc_metrics, axis=0) == 0)
                    if np.sum(intercept_mask) == 0:
                        total_qc_metrics = np.hstack((total_qc_metrics, np.ones((total_qc_metrics.shape[0],1))))
                        if total_qc_metrics_directions is not None:
                            total_qc_metrics_directions = np.append(total_qc_metrics_directions, 0)
                        intercept_mask = np.append(intercept_mask, True)

                    self.total_qc_metric_betas = np.zeros(len(intercept_mask))
                    self.total_qc_metric2_betas = np.zeros(len(intercept_mask))

                    #get univariate regression coefficients
                    (metric_beta_tildes_m, metric_ses_m, metric_z_scores_m, metric_p_values_m, metric_se_inflation_factors_m) = self._compute_beta_tildes(total_qc_metrics[pred_mask,:], z_scores[:,pred_mask], np.var(z_scores[:,pred_mask], axis=1), np.std(total_qc_metrics[pred_mask,:], axis=0), np.mean(total_qc_metrics[pred_mask,:], axis=0), resid_correlation_matrix=None, log_fun=lambda x, y=0: 1)

                    log("Mean marginal slopes are %s" % np.mean(metric_beta_tildes_m, axis=0), TRACE)

                    #filter out metrics as needed
                    keep_metrics = np.full(total_qc_metrics.shape[1], False)
                    keep_metric_inds = np.where(np.any(metric_p_values_m < 0.05, axis=0))[0]
                    keep_metrics[keep_metric_inds] = True
                    keep_metrics = np.logical_or(keep_metrics, intercept_mask)
                    if np.sum(keep_metrics) < total_qc_metrics.shape[1]:
                        log("Not using %d non-significant metrics" % (total_qc_metrics.shape[1] - np.sum(keep_metrics)))

                    if total_qc_metrics_directions is not None:
                        keep_metrics_dir = np.full(total_qc_metrics.shape[1], True)
                        keep_metric_dir_inds = np.where(np.any((metric_beta_tildes_m * total_qc_metrics_directions) < 0, axis=0))[0]
                        keep_metrics_dir[keep_metric_dir_inds] = False
                        if np.sum(keep_metrics_dir) < total_qc_metrics.shape[1]:
                            log("Not using %d metrics with wrong sign" % (total_qc_metrics.shape[1] - np.sum(keep_metrics_dir)))
                        keep_metrics = np.logical_and(keep_metrics, keep_metrics_dir)

                    total_qc_metrics_for_reg = total_qc_metrics
                    if np.sum(keep_metrics) < total_qc_metrics.shape[1]:
                        total_qc_metrics_for_reg = total_qc_metrics[:,keep_metrics]

                    total_qc_metrics_mat_inv = np.linalg.inv(total_qc_metrics_for_reg.T.dot(total_qc_metrics_for_reg))

                    pred_slopes = total_qc_metrics_mat_inv.dot(total_qc_metrics_for_reg[pred_mask,:].T).dot(z_scores[:,pred_mask].T)
                    pred2_slopes = total_qc_metrics_mat_inv.dot(total_qc_metrics_for_reg[pred_mask,:].T).dot(np.power(z_scores[:,pred_mask], 2).T)

                    #total_qc_metric_betas needs to have an entry for every metric
                    self.total_qc_metric_betas[keep_metrics] = np.mean(pred_slopes, axis=1)
                    self.total_qc_metric2_betas[keep_metrics] = np.mean(pred2_slopes, axis=1)

                    #don't use the intercept for prediction
                    #stored without the intercept
                    self.total_qc_metric_intercept = self.total_qc_metric_betas[intercept_mask]
                    self.total_qc_metric2_intercept = self.total_qc_metric2_betas[intercept_mask]
                    self.total_qc_metric_betas = self.total_qc_metric_betas[~intercept_mask]
                    self.total_qc_metric2_betas = self.total_qc_metric2_betas[~intercept_mask]

                    log("Ran regression for %d gene sets" % np.sum(pred_mask), TRACE)

                #intercept_mask = intercept_mask[keep_metrics]
                #pred_intercept = pred_slopes[intercept_mask,:]
                #pred2_intercept = pred2_slopes[intercept_mask,:]
                #pred_slopes = pred_slopes[~intercept_mask,:]
                #pred2_slopes = pred2_slopes[~intercept_mask,:]

                desired_var = np.var(z_scores, axis=1)
                self.total_qc_metric_desired_var = desired_var

                log("Mean slopes for mean are %s (+ %s)" % (self.total_qc_metric_betas, self.total_qc_metric_intercept), TRACE)
                if correct_var:
                    log("Mean slopes for square are %s (+ %s) " % (self.total_qc_metric2_betas, self.total_qc_metric2_intercept), TRACE)

                if self.gene_covariate_names is not None:
                    param_names = ["%s_beta" % self.gene_covariate_names[i] for i in range(len(self.gene_covariate_names)) if i != self.gene_covariate_intercept_index] + ["%s2_beta" % self.gene_covariate_names[i] for i in range(len(self.gene_covariate_names)) if i != self.gene_covariate_intercept_index]
                    param_values = np.append(self.total_qc_metric_betas, self.total_qc_metric2_betas)
                    self._record_params(dict(zip(param_names, param_values)), record_only_first_time=True)

            else:
                z_scores = np.zeros(beta_tildes.shape)
                z_scores[ses != 0] = np.abs(beta_tildes[ses != 0]) / ses[ses != 0]

            #pred_slopes = self.total_qc_metric_betas
            #pred2_slopes = self.total_qc_metric2_betas
            #pred_intercept = self.total_qc_metric_intercept
            #pred2_intercept = self.total_qc_metric2_intercept
            #desired_var = self.total_qc_metric_desired_var

            #adjust to (z - ax) = b + epsilon
            intercept_mask = (np.std(total_qc_metrics, axis=0) == 0)
            print(total_qc_metrics.shape)
            print(total_qc_metrics[:,~intercept_mask].shape)
            print(self.total_qc_metric_betas.shape)
            print(total_qc_metrics[:,~intercept_mask].dot(self.total_qc_metric_betas).shape)
            print(self.total_qc_metric_intercept.shape)
            pred_means = (total_qc_metrics[:,~intercept_mask].dot(self.total_qc_metric_betas) + self.total_qc_metric_intercept).T
            pred_means2 = (total_qc_metrics[:,~intercept_mask].dot(self.total_qc_metric2_betas) + self.total_qc_metric2_intercept).T
            pred_var = pred_means2 - np.square(pred_means)
            if len(pred_var.shape) == 1:
                pred_var = np.tile(pred_var, z_scores.shape[0]).reshape(z_scores.shape[0], len(pred_var))

            #first subtract out effect on E[Z] from metrics
            if correct_mean:
                pred_adjusted = ((z_scores - pred_means).T + self.total_qc_metric_intercept).T
            else:
                pred_adjusted = z_scores

            #now divide by effect on Var[Z] from metrics
            if correct_var:
                high_var_mask = np.logical_and(pred_var.T > self.total_qc_metric_desired_var, pred_var.T > 0).T
                pred_var[pred_var == 0] = 1
                variance_factors = (self.total_qc_metric_desired_var / pred_var.T).T
                pred_adjusted[high_var_mask] *= variance_factors[high_var_mask]

            #only adjust those that are predicted to decrease AND do not have zero beta tildes
            inflate_mask = np.logical_and(np.abs(pred_adjusted) < np.abs(z_scores), beta_tildes != 0)

            new_ses = copy.copy(ses)
            if np.sum(inflate_mask) > 0:
                log("Inflating %d standard errors" % (np.sum(inflate_mask)))

            new_ses[inflate_mask] = np.abs(beta_tildes[inflate_mask]) / np.abs(pred_adjusted[inflate_mask])

            if se_inflation_factors is not None:
                se_inflation_factors[inflate_mask] *= new_ses[inflate_mask] / ses[inflate_mask]

            ses = new_ses
     
        #in case original ses are zero
        zero_se_mask = ses == 0
        assert(np.sum(np.logical_and(zero_se_mask, beta_tildes != 0)) == 0)
        z_scores = np.zeros(beta_tildes.shape)
        z_scores[~zero_se_mask] = beta_tildes[~zero_se_mask] / ses[~zero_se_mask]
        p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))

        #if fit and total_qc_metrics is not None:
        #    pred_slopes_after = total_qc_metrics_mat_inv.dot(total_qc_metrics_for_reg[pred_mask,:].T).dot(z_scores[:,pred_mask].T)
        #    log("Checking new slope for %d gene sets" % np.sum(pred_mask), TRACE)
        #    log("Mean slopes after are %s" % np.mean(pred_slopes_after, axis=1), TRACE)
        #    pred_slopes_after_no_outlier = total_qc_metrics_mat_inv.dot(total_qc_metrics_for_reg.T).dot(z_scores.T)
        #    log("Mean slopes after (no outliers) are %s" % np.mean(pred_slopes_after_no_outlier, axis=1), TRACE)


        if np.sum(remove_mask) > 0:

            if correct_ignored:
                self.beta_tildes_ignored = beta_tildes[0,remove_mask]
                self.ses_ignored = ses[0,remove_mask]
                self.z_scores_ignored = z_scores[0,remove_mask]
                self.p_values_ignored = p_values[0,remove_mask]
                if se_inflation_factors is not None:
                    self.se_inflation_factors_ignored = se_inflation_factors[0,remove_mask]

            beta_tildes = beta_tildes[:,~remove_mask]
            ses = ses[:,~remove_mask]
            z_scores = z_scores[:,~remove_mask]
            p_values = p_values[:,~remove_mask]
            if se_inflation_factors is not None:
                se_inflation_factors = se_inflation_factors[:,~remove_mask]

        if beta_tildes.shape[0] == 1:
            beta_tildes = np.squeeze(beta_tildes, axis=0)
            ses = np.squeeze(ses, axis=0)
            p_values = np.squeeze(p_values, axis=0)
            z_scores = np.squeeze(z_scores, axis=0)

            if se_inflation_factors is not None:
                se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

        return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)

    def _calculate_inf_betas(self, beta_tildes=None, ses=None, V=None, V_cor=None, se_inflation_factors=None, V_inv=None, scale_factors=None, is_dense_gene_set=None):
        if V is None:
            bail("Require V")
        if beta_tildes is None:
            beta_tildes = self.beta_tildes
        if ses is None:
            ses = self.ses
        if scale_factors is None:
            scale_factors = self.scale_factors
        if is_dense_gene_set is None:
            is_dense_gene_set = self.is_dense_gene_set

        if V is None:
            bail("V is required for this operation")
        if beta_tildes is None:
            bail("Cannot calculate sigma with no stats loaded!")
        if self.sigma2 is None:
            bail("Need sigma to calculate betas!")

        log("Calculating infinitesimal betas")
        sigma2 = self.sigma2
        if self.sigma_power is not None:
            #sigma2 = self.sigma2 * np.power(scale_factors, self.sigma_power)
            sigma2 = self.get_scaled_sigma2(scale_factors, self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

            #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
            if np.sum(is_dense_gene_set) > 0:
                if np.sum(~is_dense_gene_set) > 0:
                    #sigma2[is_dense_gene_set] = self.sigma2 * np.power(np.mean(scale_factors[~is_dense_gene_set]), self.sigma_power)
                    sigma2[is_dense_gene_set] = self.get_scaled_sigma2(np.mean(scale_factors[~is_dense_gene_set]), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

                else:
                    #sigma2[is_dense_gene_set] = self.sigma2 * np.power(np.mean(scale_factors), self.sigma_power)
                    sigma2[is_dense_gene_set] = self.get_scaled_sigma2(np.mean(scale_factors), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

        orig_shrinkage_fac=np.diag(np.square(ses)/sigma2)
        shrinkage_fac = orig_shrinkage_fac

        #handle corrected OLS case
        if V_cor is not None and se_inflation_factors is not None:
            if V_inv is None:
                V_inv = self._invert_sym_matrix(V)
            shrinkage_fac = V_cor.dot(V_inv).dot(shrinkage_fac / np.square(se_inflation_factors))
            shrinkage_inv = self._invert_matrix(V + shrinkage_fac)
            return shrinkage_inv.dot(beta_tildes)
        else:
            cho_factor = scipy.linalg.cho_factor(V + shrinkage_fac)
            return scipy.linalg.cho_solve(cho_factor, beta_tildes)

    #there are two levels of parallelization here:
    #1. num_chains: sample multiple independent chains with the same beta/se/V
    #2. multiple parallel runs with different beta/se (and potentially V). To do this, pass in lists of beta and se (must be the same length) and an optional list of V (V must have same length as beta OR must be not a list, in which case the same V will be used for all betas and ses

    #to run this in parallel, pass in two-dimensional matrix for beta_tildes (rows are parallel runs, columns are beta_tildes)
    #you can pass in multiple V as well with rows/columns mapping to gene sets and a first dimension mapping to parallel runs
    def _calculate_non_inf_betas(self, initial_p, return_sample=False, max_num_burn_in=None, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, eps=0.01, max_frac_sem=0.01, max_allowed_batch_correlation=None, beta_outlier_iqr_threshold=5, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, adjust_hyper_sigma_p=False, sigma_num_devs_to_top=2.0, p_noninf_inflate=1.0, num_p_pseudo=1, sparse_solution=False, sparse_frac_betas=None, betas_trace_out=None, betas_trace_gene_sets=None, beta_tildes=None, ses=None, V=None, X_orig=None, scale_factors=None, mean_shifts=None, is_dense_gene_set=None, ps=None, sigma2s=None, assume_independent=False, num_missing_gene_sets=None, debug_genes=None, debug_gene_sets=None):

        debug_gene_sets = None

        if max_num_burn_in is None:
            max_num_burn_in = int(max_num_iter * .25)
        if max_num_burn_in >= max_num_iter:
            max_num_burn_in = int(max_num_iter * .25)

        #if (update_hyper_p or update_hyper_sigma) and gauss_seidel:
        #    log("Using Gibbs sampling for betas since update hyper was requested")
        #    gauss_seidel = False

        if ses is None:
            ses = self.ses
        if beta_tildes is None:
            beta_tildes = self.beta_tildes
            
        if X_orig is None and not assume_independent:
            X_orig = self.X_orig
        if scale_factors is None:
            scale_factors = self.scale_factors
        if mean_shifts is None:
            mean_shifts = self.mean_shifts

        use_X = False
        if V is None and not assume_independent:
            if X_orig is None or scale_factors is None or mean_shifts is None:
                bail("Require X, scale, and mean if V is None")
            else:
                use_X = True
                log("Using low memory X instead of V", TRACE)

        if is_dense_gene_set is None:
            is_dense_gene_set = self.is_dense_gene_set
        if ps is None:
            ps = self.ps
        if sigma2s is None:
            sigma2s = self.sigma2s

        if self.sigma2 is None:
            bail("Need sigma to calculate betas!")

        if initial_p is not None:
            self.set_p(initial_p)

        if self.p is None and ps is None:
            bail("Need p to calculate non-inf betas")

        if not len(beta_tildes.shape) == len(ses.shape):
            bail("If running parallel beta inference, beta_tildes and ses must have same shape")

        if len(beta_tildes.shape) == 0 or beta_tildes.shape[0] == 0:
            bail("No gene sets are left!")

        #convert the beta_tildes and ses to matrices -- columns are num_parallel
        #they are always stored as matrices, with 1 column as needed
        #V on the other hand will be a 2-D matrix if it is constant across all parallel (or if there is only 1)
        #checking len(V.shape) can therefore distinguish a constant from variable V

        multiple_V = False
        sparse_V = False

        if len(beta_tildes.shape) > 1:
            num_gene_sets = beta_tildes.shape[1]

            if not beta_tildes.shape[0] == ses.shape[0]:
                bail("beta_tildes and ses must have same number of parallel runs")

            #dimensions should be num_gene_sets, num_parallel
            num_parallel = beta_tildes.shape[0]
            beta_tildes_m = copy.copy(beta_tildes)
            ses_m = copy.copy(ses)

            if V is not None and type(V) is sparse.csc_matrix:
                sparse_V = True
                multiple_V = False
            elif V is not None and len(V.shape) == 3:
                if not V.shape[0] == beta_tildes.shape[0]:
                    bail("V must have same number of parallel runs as beta_tildes")
                multiple_V = True
                sparse_V = False
            else:
                multiple_V = False
                sparse_V = False

        else:
            num_gene_sets = len(beta_tildes)
            if V is not None and type(V) is sparse.csc_matrix:
                num_parallel = 1
                multiple_V = False
                sparse_V = True
                beta_tildes_m = beta_tildes[np.newaxis,:]
                ses_m = ses[np.newaxis,:]
            elif V is not None and len(V.shape) == 3:
                num_parallel = V.shape[0]
                multiple_V = True
                sparse_V = False
                beta_tildes_m = np.tile(beta_tildes, num_parallel).reshape((num_parallel, len(beta_tildes)))
                ses_m = np.tile(ses, num_parallel).reshape((num_parallel, len(ses)))
            else:
                num_parallel = 1
                multiple_V = False
                sparse_V = False
                beta_tildes_m = beta_tildes[np.newaxis,:]
                ses_m = ses[np.newaxis,:]

        if num_parallel == 1 and multiple_V:
            multiple_V = False
            V = V[0,:,:]

        if multiple_V:
            assert(not use_X)

        if scale_factors.shape != mean_shifts.shape:
            bail("scale_factors must have same dimension as mean_shifts")

        if len(scale_factors.shape) == 2 and not scale_factors.shape[0] == num_parallel:
            bail("scale_factors must have same number of parallel runs as beta_tildes")
        elif len(scale_factors.shape) == 1 and num_parallel == 1:
            scale_factors_m = scale_factors[np.newaxis,:]
            mean_shifts_m = mean_shifts[np.newaxis,:]
        elif len(scale_factors.shape) == 1 and num_parallel > 1:
            scale_factors_m = np.tile(scale_factors, num_parallel).reshape((num_parallel, len(scale_factors)))
            mean_shifts_m = np.tile(mean_shifts, num_parallel).reshape((num_parallel, len(mean_shifts)))
        else:
            scale_factors_m = copy.copy(scale_factors)
            mean_shifts_m = copy.copy(mean_shifts)

        if len(is_dense_gene_set.shape) == 2 and not is_dense_gene_set.shape[0] == num_parallel:
            bail("is_dense_gene_set must have same number of parallel runs as beta_tildes")
        elif len(is_dense_gene_set.shape) == 1 and num_parallel == 1:
            is_dense_gene_set_m = is_dense_gene_set[np.newaxis,:]
        elif len(is_dense_gene_set.shape) == 1 and num_parallel > 1:
            is_dense_gene_set_m = np.tile(is_dense_gene_set, num_parallel).reshape((num_parallel, len(is_dense_gene_set)))
        else:
            is_dense_gene_set_m = copy.copy(is_dense_gene_set)

        if ps is not None:
            if len(ps.shape) == 2 and not ps.shape[0] == num_parallel:
                bail("ps must have same number of parallel runs as beta_tildes")
            elif len(ps.shape) == 1 and num_parallel == 1:
                ps_m = ps[np.newaxis,:]
            elif len(ps.shape) == 1 and num_parallel > 1:
                ps_m = np.tile(ps, num_parallel).reshape((num_parallel, len(ps)))
            else:
                ps_m = copy.copy(ps)
        else:
            ps_m = self.p

        if sigma2s is not None:
            if len(sigma2s.shape) == 2 and not sigma2s.shape[0] == num_parallel:
                bail("sigma2s must have same number of parallel runs as beta_tildes")
            elif len(sigma2s.shape) == 1 and num_parallel == 1:
                orig_sigma2_m = sigma2s[np.newaxis,:]
            elif len(sigma2s.shape) == 1 and num_parallel > 1:
                orig_sigma2_m = np.tile(sigma2s, num_parallel).reshape((num_parallel, len(sigma2s)))
            else:
                orig_sigma2_m = copy.copy(sigma2s)
        else:
            orig_sigma2_m = self.sigma2

        #for efficiency, batch genes to be updated each cycle
        if assume_independent:
            gene_set_masks = [np.full(beta_tildes_m.shape[1], True)]
        else:
            gene_set_masks = self._compute_gene_set_batches(V, X_orig=X_orig, mean_shifts=mean_shifts, scale_factors=scale_factors, use_sum=True, max_allowed_batch_correlation=max_allowed_batch_correlation)
            
        sizes = [float(np.sum(x)) / (num_parallel if multiple_V else 1) for x in gene_set_masks]
        log("Analyzing %d gene sets in %d batches of gene sets; size range %d - %d" % (num_gene_sets, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes)  if len(sizes) > 0 else 0), DEBUG)

        #get the dimensions of the gene_set_masks to match those of the betas
        if num_parallel == 1:
            assert(not multiple_V)
            #convert the vectors into matrices with one dimension
            gene_set_masks = [x[np.newaxis,:] for x in gene_set_masks]
        elif not multiple_V:
            #we have multiple parallel but only one V
            gene_set_masks = [np.tile(x, num_parallel).reshape((num_parallel, len(x))) for x in gene_set_masks]

        #variables are denoted
        #v: vectors of dimension equal to the number of gene sets
        #m: data that varies by parallel runs and gene sets
        #t: data that varies by chains, parallel runs, and gene sets

        #rules:
        #1. adding a lower dimensional tensor to higher dimenional ones means final dimensions must match. These operations are usually across replicates
        #2. lower dimensional masks on the other hand index from the beginning dimensions (can use :,:,mask to index from end)
        
        tensor_shape = (num_chains, num_parallel, num_gene_sets)
        matrix_shape = (num_parallel, num_gene_sets)

        #these are current posterior means (including p and the conditional beta). They are used to calculate avg_betas
        #using these as the actual betas would yield the Gauss-seidel algorithm
        curr_post_means_t = np.zeros(tensor_shape)
        curr_postp_t = np.ones(tensor_shape)

        #these are the current betas to be used in each iteration
        initial_sd = np.std(beta_tildes_m)
        if initial_sd == 0:
            initial_sd = 1

        curr_betas_t = scipy.stats.norm.rvs(0, initial_sd, tensor_shape)

        res_beta_hat_t = np.zeros(tensor_shape)

        avg_betas_m = np.zeros(matrix_shape)
        avg_betas2_m = np.zeros(matrix_shape)
        avg_postp_m = np.zeros(matrix_shape)
        num_avg = 0

        #these are the posterior betas averaged across iterations
        sum_betas_t = np.zeros(tensor_shape)
        sum_betas2_t = np.zeros(tensor_shape)

        # Setting up constants
        #hyperparameters
        #shrinkage prior
        if self.sigma_power is not None:
            #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
            sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

            #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
            if np.sum(is_dense_gene_set_m) > 0:
                if np.sum(~is_dense_gene_set_m) > 0:
                    #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
                    sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                else:
                    #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
                    sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

        else:
            sigma2_m = orig_sigma2_m

        if ps_m is not None and np.min(ps_m) != np.max(ps_m):
            p_text = "mean p=%.3g (%.3g-%.3g)" % (self.p, np.min(ps_m), np.max(ps_m))
        else:
            p_text = "p=%.3g" % (self.p)
        if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
            sigma2_text = "mean sigma=%.3g (%.3g-%.3g)" % (self.sigma2, np.min(orig_sigma2_m), np.max(orig_sigma2_m))
        else:
            sigma2_text = "sigma=%.3g" % (self.sigma2)

        if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
            sigma2_p_text = "mean sigma2/p=%.3g (%.3g-%.3g)" % (self.sigma2/self.p, np.min(orig_sigma2_m/ps_m), np.max(orig_sigma2_m/ps_m))
        else:
            sigma2_p_text = "sigma2/p=%.3g" % (self.sigma2/self.p)


        tag = ""
        if assume_independent:
            tag = "independent "
        elif sparse_V:
            tag = "partially independent "
            
        log("Calculating %snon-infinitesimal betas with %s, %s; %s" % (tag, p_text, sigma2_text, sigma2_p_text))

        #generate the diagonals to use per replicate
        if assume_independent:
            V_diag_m = None
            account_for_V_diag_m = False
        else:
            if V is not None:
                if num_parallel > 1:
                    #dimensions are num_parallel, num_gene_sets, num_gene_sets
                    if multiple_V:
                        V_diag_m = np.diagonal(V, axis1=1, axis2=2)
                    else:
                        if sparse_V:
                            V_diag = V.diagonal()
                        else:
                            V_diag = np.diag(V)
                        V_diag_m = np.tile(V_diag, num_parallel).reshape((num_parallel, len(V_diag)))
                else:
                    if sparse_V:
                        V_diag_m = V.diagonal()[np.newaxis,:]                        
                    else:
                        V_diag_m = np.diag(V)[np.newaxis,:]

                account_for_V_diag_m = not np.isclose(V_diag_m, np.ones(matrix_shape)).all()
            else:
                #we compute it from X, so we know it is always 1
                V_diag_m = None
                account_for_V_diag_m = False

        se2s_m = np.power(ses_m,2)

        #the below code is based off of the LD-pred code for SNP PRS
        iteration_num = 0
        burn_in_phase_v = np.array([True for i in range(num_parallel)])


        if betas_trace_out is not None:
            betas_trace_fh = open_gz(betas_trace_out, 'w')
            betas_trace_fh.write("It\tParallel\tChain\tGene_Set\tbeta_post\tbeta\tpostp\tres_beta_hat\tbeta_tilde\tbeta_internal\tres_beta_hat_internal\tbeta_tilde_internal\tse_internal\tsigma2\tp\tR\tR_weighted\tSEM\n")

        prev_betas_m = None
        sigma_underflow = False
        printed_warning_swing = False
        printed_warning_increase = False
        while iteration_num < max_num_iter:  #Big iteration

            #if some have not converged, only sample for those that have not converged (for efficiency)
            compute_mask_v = copy.copy(burn_in_phase_v)
            if np.sum(compute_mask_v) == 0:
                compute_mask_v[:] = True

            hdmp_m = (sigma2_m / ps_m)
            hdmpn_m = hdmp_m + se2s_m
            hdmp_hdmpn_m = (hdmp_m / hdmpn_m)

            norm_scale_m = np.sqrt(np.multiply(hdmp_hdmpn_m, se2s_m))
            c_const_m = (ps_m / np.sqrt(hdmpn_m))

            d_const_m = (1 - ps_m) / ses_m

            iteration_num += 1

            #default to 1
            curr_postp_t[:,compute_mask_v,:] = np.ones(tensor_shape)[:,compute_mask_v,:]

            #sample whether each gene set has non-zero effect
            rand_ps_t = np.random.random(tensor_shape)
            #generate normal random variable sampling
            rand_norms_t = scipy.stats.norm.rvs(0, 1, tensor_shape)

            for gene_set_mask_ind in range(len(gene_set_masks)):

                #the challenge here is that gene_set_mask_m produces a ragged (non-square) tensor
                #so we are going to "flatten" the last two dimensions
                #this requires some care, in particular when running einsum, which requires a square tensor

                gene_set_mask_m = gene_set_masks[gene_set_mask_ind]
                
                if debug_gene_sets is not None:
                    cur_debug_gene_sets = [debug_gene_sets[i] for i in range(len(debug_gene_sets)) if gene_set_mask_m[0,i]]

                #intersect compute_max_v with the rows of gene_set_mask (which are the parallel runs)
                compute_mask_m = np.logical_and(compute_mask_v, gene_set_mask_m.T).T

                current_num_parallel = sum(compute_mask_v)

                #Value to use when determining if we should force an alpha shrink if estimates are way off compared to heritability estimates.  (Improves MCMC convergence.)
                #zero_jump_prob=0.05
                #frac_betas_explained = max(0.00001,np.sum(np.apply_along_axis(np.mean, 0, np.power(curr_betas_m,2)))) / self.y_var
                #frac_sigma_explained = self.sigma2_total_var / self.y_var
                #alpha_shrink = min(1 - zero_jump_prob, 1.0 / frac_betas_explained, (frac_sigma_explained + np.mean(np.power(ses[i], 2))) / frac_betas_explained)
                alpha_shrink = 1

                #subtract out the predicted effects of the other betas
                #we need to zero out diagonal of V to do this, but rather than do this we will add it back in

                #1. First take the union of the current_gene_set_mask
                #this is to allow us to run einsum
                #we are going to do it across more gene sets than are needed, and then throw away the computations that are extra for each batch
                compute_mask_union = np.any(compute_mask_m, axis=0)

                #2. Retain how to filter from the union down to each mask
                compute_mask_union_filter_m = compute_mask_m[:,compute_mask_union]

                if assume_independent:
                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m]
                else:
                    if multiple_V:

                        #3. Do einsum across the union
                        #This does pointwise matrix multiplication of curr_betas_t (sliced on axis 1) with V (sliced on axis 0), maintaining axis 0 for curr_betas_t
                        res_beta_hat_union_t = np.einsum('hij,ijk->hik', curr_betas_t[:,compute_mask_v,:], V[compute_mask_v,:,:][:,:,compute_mask_union]).reshape((num_chains, current_num_parallel, np.sum(compute_mask_union)))

                    elif sparse_V:
                        res_beta_hat_union_t = V[compute_mask_union,:].dot(curr_betas_t[:,compute_mask_v,:].T.reshape((curr_betas_t.shape[2], np.sum(compute_mask_v) * curr_betas_t.shape[0]))).reshape((np.sum(compute_mask_union), np.sum(compute_mask_v), curr_betas_t.shape[0])).T
                    elif use_X:
                        if len(compute_mask_union.shape) == 2:
                            assert(compute_mask_union.shape[0] == 1)
                            compute_mask_union = np.squeeze(compute_mask_union)
                        #curr_betas_t: (num_chains, num_parallel, num_gene_sets)
                        #X_orig: (num_genes, num_gene_sets)
                        #X_orig_t: (num_gene_sets, num_genes)
                        #mean_shifts_m: (num_parallel, num_gene_sets)
                        #curr_betas_filtered_t: (num_chains, num_compute, num_gene_sets)

                        curr_betas_filtered_t = curr_betas_t[:,compute_mask_v,:] / scale_factors_m[compute_mask_v,:]

                        #have to reshape latter two dimensions before multiplying because sparse matrix can only handle 2-D

                        #interm = np.zeros((X_orig.shape[0],np.sum(compute_mask_v),curr_betas_t.shape[0]))
                        #interm[:,compute_mask_v,:] = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1]))).reshape((X_orig.shape[0],curr_betas_filtered_t.shape[1],curr_betas_filtered_t.shape[0])) - np.sum(mean_shifts_m[compute_mask_v,:] * curr_betas_filtered_t, axis=2).T

                        interm = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1]))).reshape((X_orig.shape[0],curr_betas_filtered_t.shape[1],curr_betas_filtered_t.shape[0])) - np.sum(mean_shifts_m[compute_mask_v,:] * curr_betas_filtered_t, axis=2).T

                        #interm: (num_genes, num_parallel remaining, num_chains)

                        #num_gene sets, num_parallel, num_chains

                        #this broke under some circumstances when a parallel chain converged before the others
                        res_beta_hat_union_t = (X_orig[:,compute_mask_union].T.dot(interm.reshape((interm.shape[0],interm.shape[1]*interm.shape[2]))).reshape((np.sum(compute_mask_union),interm.shape[1],interm.shape[2])) - mean_shifts_m.T[compute_mask_union,:][:,compute_mask_v,np.newaxis] * np.sum(interm, axis=0)).T
                        res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[compute_mask_v,:][:,compute_mask_union])

                        #res_beta_hat_union_t = (X_orig[:,compute_mask_union].T.dot(interm.reshape((interm.shape[0],interm.shape[1]*interm.shape[2]))).reshape((np.sum(compute_mask_union),interm.shape[1],interm.shape[2])) - mean_shifts_m.T[compute_mask_union,:][:,:,np.newaxis] * np.sum(interm, axis=0)).T
                        #res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[:,compute_mask_union])

                    else:
                        res_beta_hat_union_t = curr_betas_t[:,compute_mask_v,:].dot(V[:,compute_mask_union])

                    if betas_trace_out is not None and betas_trace_gene_sets is not None:
                        all_map = self._construct_map_to_ind(betas_trace_gene_sets)
                        cur_sets = [betas_trace_gene_sets[x] for x in range(len(betas_trace_gene_sets)) if compute_mask_union[x]]
                        cur_map = self._construct_map_to_ind(cur_sets)

                    #4. Now restrict to only the actual masks (which flattens things because the compute_mask_m is not square)

                    res_beta_hat_t_flat = res_beta_hat_union_t[:,compute_mask_union_filter_m[compute_mask_v,:]]
                    assert(res_beta_hat_t_flat.shape[1] == np.sum(compute_mask_m))

                    #dimensions of res_beta_hat_t_flat are (num_chains, np.sum(compute_mask_m))
                    #dimensions of beta_tildes_m are (num_parallel, num_gene_sets))
                    #subtraction will subtract matrix from each of the matrices in the tensor

                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m] - res_beta_hat_t_flat

                    if account_for_V_diag_m:
                        #dimensions of V_diag_m are (num_parallel, num_gene_sets)
                        #curr_betas_t is (num_chains, num_parallel, num_gene_sets)
                        res_beta_hat_t_flat = res_beta_hat_t_flat + V_diag_m[compute_mask_m] * curr_betas_t[:,compute_mask_m]
                    else:
                        res_beta_hat_t_flat = res_beta_hat_t_flat + curr_betas_t[:,compute_mask_m]
                
                b2_t_flat = np.power(res_beta_hat_t_flat, 2)
                d_const_b2_exp_t_flat = d_const_m[compute_mask_m] * np.exp(-b2_t_flat / (se2s_m[compute_mask_m] * 2.0))
                numerator_t_flat = c_const_m[compute_mask_m] * np.exp(-b2_t_flat / (2.0 * hdmpn_m[compute_mask_m]))
                numerator_zero_mask_t_flat = (numerator_t_flat == 0)
                denominator_t_flat = numerator_t_flat + d_const_b2_exp_t_flat
                denominator_t_flat[numerator_zero_mask_t_flat] = 1


                d_imaginary_mask_t_flat = ~np.isreal(d_const_b2_exp_t_flat)
                numerator_imaginary_mask_t_flat = ~np.isreal(numerator_t_flat)

                if np.any(np.logical_or(d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)):

                    warn("Detected imaginary numbers!")
                    #if d is imaginary, we set it to 1
                    denominator_t_flat[d_imaginary_mask_t_flat] = numerator_t_flat[d_imaginary_mask_t_flat]
                    #if d is real and numerator is imaginary, we set to 0 (both numerator and denominator will be imaginary)
                    numerator_t_flat[np.logical_and(~d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)] = 0

                    #Original code for handling edge cases; adapted above
                    #Commenting these out for now, but they are here in case we ever detect non real numbers
                    #if need them, masked_array is too inefficient -- change to real mask
                    #d_real_mask_t = np.isreal(d_const_b2_exp_t)
                    #numerator_real_mask_t = np.isreal(numerator_t)
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_not(d_real_mask_t), fill_value = 1).filled()
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(d_real_mask_t, np.logical_not(numerator_real_mask_t)), fill_value=0).filled()
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(np.logical_and(d_real_mask_t, numerator_real_mask_t), numerator_zero_mask_t), fill_value=0).filled()



                curr_postp_t[:,compute_mask_m] = (numerator_t_flat / denominator_t_flat)


                #calculate current posterior means
                #the left hand side, because it is masked, flattens the latter two dimensions into one
                #so we flatten the result of the right hand size to a 1-D array to match up for the assignment
                curr_post_means_t[:,compute_mask_m] = hdmp_hdmpn_m[compute_mask_m] * (curr_postp_t[:,compute_mask_m] * res_beta_hat_t_flat)

                   
                if gauss_seidel:
                    proposed_beta_t_flat = curr_post_means_t[:,compute_mask_m]
                else:
                    norm_mean_t_flat = hdmp_hdmpn_m[compute_mask_m] * res_beta_hat_t_flat

                    #draw from the conditional distribution
                    proposed_beta_t_flat = norm_mean_t_flat + norm_scale_m[compute_mask_m] * rand_norms_t[:,compute_mask_m]

                    #set things to zero that sampled below p
                    zero_mask_t_flat = rand_ps_t[:,compute_mask_m] >= curr_postp_t[:,compute_mask_m] * alpha_shrink
                    proposed_beta_t_flat[zero_mask_t_flat] = 0

                #update betas
                #do this inside loop since this determines the res_beta
                #same idea as above for collapsing
                curr_betas_t[:,compute_mask_m] = proposed_beta_t_flat
                res_beta_hat_t[:,compute_mask_m] = res_beta_hat_t_flat

                #if debug_gene_sets is not None:
                #    my_cur_tensor_shape = (1 if assume_independent else num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
                #    my_cur_tensor_shape2 = (num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
                #    my_res_beta_hat_t = res_beta_hat_t_flat.reshape(my_cur_tensor_shape)
                #    my_proposed_beta_t = proposed_beta_t_flat.reshape(my_cur_tensor_shape2)
                #    my_norm_mean_t = norm_mean_t_flat.reshape(my_cur_tensor_shape)
                #    top_set = [cur_debug_gene_sets[i] for i in range(len(cur_debug_gene_sets)) if np.abs(my_res_beta_hat_t[0,0,i]) == np.max(np.abs(my_res_beta_hat_t[0,0,:]))][0]
                #    log("TOP IS",top_set)
                #    gs = set([ "mp_absent_T_cells", top_set])
                #    ind = [i for i in range(len(cur_debug_gene_sets)) if cur_debug_gene_sets[i] in gs]
                #    for i in ind:
                #        log("BETA_TILDE",cur_debug_gene_sets[i],beta_tildes_m[0,i]/scale_factors_m[0,i])
                #        log("Z",cur_debug_gene_sets[i],beta_tildes_m[0,i]/ses_m[0,i])
                #        log("RES",cur_debug_gene_sets[i],my_res_beta_hat_t[0,0,i]/scale_factors_m[0,i])
                #        #log("RESF",cur_debug_gene_sets[i],res_beta_hat_t_flat[i]/scale_factors_m[0,i])
                #        log("NORM_MEAN",cur_debug_gene_sets[i],my_norm_mean_t[0,0,i])
                #        log("NORM_SCALE_M",cur_debug_gene_sets[i],norm_scale_m[0,i])
                #        log("RAND_NORMS",cur_debug_gene_sets[i],rand_norms_t[0,0,i])
                #        log("PROP",cur_debug_gene_sets[i],my_proposed_beta_t[0,0,i]/scale_factors_m[0,i])
                #        ind2 = [j for j in range(len(debug_gene_sets)) if debug_gene_sets[j] == cur_debug_gene_sets[i]]
                #        for j in ind2:
                #            log("POST",cur_debug_gene_sets[i],curr_post_means_t[0,0,j]/scale_factors_m[0,i])
                #            log("SIGMA",sigma2_m if type(sigma2_m) is float or type(sigma2_m) is np.float64 else sigma2_m[0,i])
                #            log("P",cur_debug_gene_sets[i],curr_postp_t[0,0,j],self.p)
                #            log("HDMP",hdmp_m/np.square(scale_factors_m[0,i]) if type(hdmp_m) is float or type(hdmp_m) is np.float64 else hdmp_m[0,0]/np.square(scale_factors_m[0,i]))
                #            log("SES",se2s_m[0,0]/np.square(scale_factors_m[0,i]))
                #            log("HDMPN",hdmpn_m/np.square(scale_factors_m[0,i]) if type(hdmpn_m) is float or type(hdmpn_m) is np.float64 else hdmpn_m[0,0]/scale_factors_m[0,i])
                #            log("HDMP_HDMPN",hdmp_hdmpn_m if type(hdmp_hdmpn_m) is float or type(hdmp_hdmpn_m) is np.float64 else hdmp_hdmpn_m[0,0])
                #            log("NOW1",debug_gene_sets[j],curr_betas_t[0,0,j]/scale_factors_m[0,i])


            if sparse_solution:
                sparse_mask_t = curr_postp_t < ps_m

                if sparse_frac_betas is not None:
                    #zero out very small values relative to top or median
                    relative_value = np.max(np.abs(curr_post_means_t), axis=2)
                    sparse_mask_t = np.logical_or(sparse_mask_t, (np.abs(curr_post_means_t).T < sparse_frac_betas * relative_value.T).T)

                #don't set anything not currently computed
                sparse_mask_t[:,np.logical_not(compute_mask_v),:] = False
                log("Setting %d entries to zero due to sparsity" % (np.sum(np.logical_and(sparse_mask_t, curr_betas_t > 0))), TRACE)
                curr_betas_t[sparse_mask_t] = 0
                curr_post_means_t[sparse_mask_t] = 0

                if debug_gene_sets is not None:
                    ind = [i for i in range(len(debug_gene_sets)) if debug_gene_sets[i] in gs]

            curr_betas_m = np.mean(curr_post_means_t, axis=0)
            curr_postp_m = np.mean(curr_postp_t, axis=0)
            #no state should be preserved across runs, but take a random one just in case
            sample_betas_m = curr_betas_t[int(random.random() * curr_betas_t.shape[0]),:,:]
            sample_postp_m = curr_postp_t[int(random.random() * curr_postp_t.shape[0]),:,:]
            sum_betas_t[:,compute_mask_v,:] = sum_betas_t[:,compute_mask_v,:] + curr_post_means_t[:,compute_mask_v,:]
            sum_betas2_t[:,compute_mask_v,:] = sum_betas2_t[:,compute_mask_v,:] + np.square(curr_post_means_t[:,compute_mask_v,:])

            #now calculate the convergence metrics
            R_m = np.zeros(matrix_shape)
            beta_weights_m = np.zeros(matrix_shape)
            sem2_m = np.zeros(matrix_shape)
            will_break = False
            if assume_independent:
                burn_in_phase_v[:] = False
            elif gauss_seidel:
                if prev_betas_m is not None:
                    sum_diff = np.sum(np.abs(prev_betas_m - curr_betas_m))
                    sum_prev = np.sum(np.abs(prev_betas_m))
                    tot_diff = sum_diff / sum_prev
                    log("Iteration %d: gauss seidel difference = %.4g / %.4g = %.4g" % (iteration_num+1, sum_diff, sum_prev, tot_diff), TRACE)
                    if iteration_num > min_num_iter and tot_diff < eps:
                        burn_in_phase_v[:] = False
                        log("Converged after %d iterations" % (iteration_num+1), INFO)
                prev_betas_m = curr_betas_m
            elif iteration_num > min_num_iter and np.sum(burn_in_phase_v) > 0:
                def __calculate_R_tensor(sum_t, sum2_t, num):

                    #mean of betas across all iterations; psi_dot_j
                    mean_t = sum_t / float(num)

                    #mean of betas across replicates; psi_dot_dot
                    mean_m = np.mean(mean_t, axis=0)
                    #variances of betas across all iterators; s_j
                    var_t = (sum2_t - float(num) * np.power(mean_t, 2)) / (float(num) - 1)
                    #B_v = (float(iteration_num) / (num_chains - 1)) * np.apply_along_axis(np.sum, 0, np.apply_along_axis(lambda x: np.power(x - mean_betas_v, 2), 1, mean_betas_m))
                    B_m = (float(num) / (mean_t.shape[0] - 1)) * np.sum(np.power(mean_t - mean_m, 2), axis=0)
                    W_m = (1.0 / float(mean_t.shape[0])) * np.sum(var_t, axis=0)
                    avg_W_m = (1.0 / float(mean_t.shape[2])) * np.sum(var_t, axis=2)
                    var_given_y_m = np.add((float(num) - 1) / float(num) * W_m, (1.0 / float(num)) * B_m)
                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m = np.ones(W_m.shape)
                    R_non_zero_mask_m = W_m > 0

                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m[R_non_zero_mask_m] = np.sqrt(var_given_y_m[R_non_zero_mask_m] / W_m[R_non_zero_mask_m])
                    
                    return (B_m, W_m, R_m, avg_W_m, mean_t)

                #these matrices have convergence statistics in format (num_parallel, num_gene_sets)
                #WARNING: only the results for compute_mask_v are valid
                (B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num)

                beta_weights_m = np.zeros((sum_betas_t.shape[1], sum_betas_t.shape[2]))
                sum_betas_t_mean = np.mean(sum_betas_t)
                if sum_betas_t_mean > 0:
                    np.mean(sum_betas_t, axis=0) / sum_betas_t_mean

                #calculate the thresholded / scaled R_v
                num_R_above_1_v = np.sum(R_m >= 1, axis=1)
                num_R_above_1_v[num_R_above_1_v == 0] = 1

                #mean for each parallel run

                R_m_above_1 = copy.copy(R_m)
                R_m_above_1[R_m_above_1 < 1] = 0
                mean_thresholded_R_v = np.sum(R_m_above_1, axis=1) / num_R_above_1_v

                #max for each parallel run
                max_index_v = np.argmax(R_m, axis=1)
                max_index_parallel = None
                max_val = None
                for i in range(len(max_index_v)):
                    if compute_mask_v[i] and (max_val is None or R_m[i,max_index_v[i]] > max_val):
                        max_val = R_m[i,max_index_v[i]]
                        max_index_parallel = i
                max_R_v = np.max(R_m, axis=1)
               

                #TEMP TEMP TEMP
                #if priors_for_convergence:
                #    curr_v = curr_betas_v
                #    s_cur2_v = np.array([curr_v[i] for i in sorted(range(len(curr_v)), key=lambda k: -np.abs(curr_v[k]))])
                #    s_cur2_v = np.square(s_cur2_v - np.mean(s_cur2_v))
                #    cum_cur2_v = np.cumsum(s_cur2_v) / np.sum(s_cur2_v)
                #    top_mask2 = np.array(cum_cur2_v < 0.99)
                #    (B_v2, W_v2, R_v2) = __calculate_R(sum_betas_m[:,top_mask2], sum_betas2_m[:,top_mask2], iteration_num)
                #    max_index2 = np.argmax(R_v2)
                #    log("Iteration %d (betas): max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g" % (iteration_num, max_index2, B_v2[max_index2], W_v2[max_index2], R_v2[max_index2], np.mean(R_v2), np.sum(R_v2 > r_threshold_burn_in)), TRACE)
                #END TEMP TEMP TEMP
                    
                if use_max_r_for_convergence:
                    convergence_statistic_v = max_R_v
                else:
                    convergence_statistic_v = mean_thresholded_R_v

                outlier_mask_m = np.full(avg_W_m.shape, False)
                if avg_W_m.shape[0] > 10:
                    #check the variances
                    q3, median, q1 = np.percentile(avg_W_m, [75, 50, 25], axis=0)
                    iqr_mask = q3 > q1
                    chain_iqr_m = np.zeros(avg_W_m.shape)
                    chain_iqr_m[:,iqr_mask] = (avg_W_m[:,iqr_mask] - median[iqr_mask]) / (q3 - q1)[iqr_mask]
                    #dimensions chain x parallel
                    outlier_mask_m = beta_outlier_iqr_threshold
                    if np.sum(outlier_mask_m) > 0:
                        log("Detected %d outlier chains due to oscillations" % np.sum(outlier_mask_m), DEBUG)

                if np.sum(R_m > 1) > 10:
                    #check the Rs
                    q3, median, q1 = np.percentile(R_m[R_m > 1], [75, 50, 25])
                    if q3 > q1:
                        #Z score per parallel, gene
                        R_iqr_m = (R_m - median) / (q3 - q1)
                        #dimensions of parallel x gene sets
                        bad_gene_sets_m = np.logical_and(R_iqr_m > 100, R_m > 2.5)
                        bad_gene_sets_v = np.any(bad_gene_sets_m,0)
                        if np.sum(bad_gene_sets_m) > 0:
                            #now find the bad chains
                            bad_chains = np.argmax(np.abs(mean_t - np.mean(mean_t, axis=0)), axis=0)[bad_gene_sets_m]

                            #np.where bad gene sets[0] lists parallel
                            #bad chains lists the bad chain corresponding to each parallel
                            cur_outlier_mask_m = np.zeros(outlier_mask_m.shape)
                            cur_outlier_mask_m[bad_chains, np.where(bad_gene_sets_m)[0]] = True

                            log("Found %d outlier chains across %d parallel runs due to %d gene sets with high R (%.4g - %.4g; %.4g - %.4g)" % (np.sum(cur_outlier_mask_m), np.sum(np.any(cur_outlier_mask_m, axis=0)), np.sum(bad_gene_sets_m), np.min(R_m[bad_gene_sets_m]), np.max(R_m[bad_gene_sets_m]), np.min(R_iqr_m[bad_gene_sets_m]), np.max(R_iqr_m[bad_gene_sets_m])), DEBUG)
                            outlier_mask_m = np.logical_or(outlier_mask_m, cur_outlier_mask_m)

                            #log("Outlier parallel: %s" % (np.where(bad_gene_sets_m)[0]), DEBUG)
                            #log("Outlier values: %s" % (R_m[bad_gene_sets_m]), DEBUG)
                            #log("Outlier IQR: %s" % (R_iqr_m[bad_gene_sets_m]), DEBUG)
                            #log("Outlier chains: %s" % (bad_chains), DEBUG)


                            #log("Actually in mask: %s" % (str(np.where(outlier_mask_m))))

                non_outliers_m = ~outlier_mask_m
                if np.sum(outlier_mask_m) > 0:
                    log("Detected %d total outlier chains" % np.sum(outlier_mask_m), DEBUG)
                    #dimensions are num_chains x num_parallel
                    for outlier_parallel in np.where(np.any(outlier_mask_m, axis=0))[0]:
                        #find a non-outlier chain and replace the three matrices in the right place
                        if np.sum(outlier_mask_m[:,outlier_parallel]) > 0:
                            if np.sum(non_outliers_m[:,outlier_parallel]) > 0:
                                replacement_chains = np.random.choice(np.where(non_outliers_m[:,outlier_parallel])[0], size=np.sum(outlier_mask_m[:,outlier_parallel]))
                                log("Replaced chains %s with chains %s in parallel %d" % (np.where(outlier_mask_m[:,outlier_parallel])[0], replacement_chains, outlier_parallel), DEBUG)

                                for tensor in [curr_betas_t, curr_postp_t, curr_post_means_t, sum_betas_t, sum_betas2_t]:
                                    tensor[outlier_mask_m[:,outlier_parallel],outlier_parallel,:] = copy.copy(tensor[replacement_chains,outlier_parallel,:])

                            else:
                                log("Every chain was an outlier so doing nothing", TRACE)


                log("Iteration %d: max ind=%s; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (iteration_num, (max_index_parallel, max_index_v[max_index_parallel]) if num_parallel > 1 else max_index_v[max_index_parallel], B_m[max_index_parallel, max_index_v[max_index_parallel]], W_m[max_index_parallel, max_index_v[max_index_parallel]], R_m[max_index_parallel, max_index_v[max_index_parallel]], np.mean(mean_thresholded_R_v), np.sum(R_m > r_threshold_burn_in)), TRACE)

                converged_v = convergence_statistic_v < r_threshold_burn_in
                newly_converged_v = np.logical_and(burn_in_phase_v, converged_v)
                if np.sum(newly_converged_v) > 0:
                    if num_parallel == 1:
                        log("Converged after %d iterations" % iteration_num, INFO)
                    else:
                        log("Parallel %s converged after %d iterations" % (",".join([str(p) for p in np.nditer(np.where(newly_converged_v))]), iteration_num), INFO)
                    burn_in_phase_v = np.logical_and(burn_in_phase_v, np.logical_not(converged_v))

            if np.sum(burn_in_phase_v) == 0 or iteration_num >= max_num_burn_in:

                if return_sample:

                    frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
                    if frac_increase > 0.01:
                        warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
                        printed_warning_increase = True

                    frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
                    if frac_opposite > 0.01:
                        warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
                        printed_warning_swing = False

                    if np.sum(burn_in_phase_v) > 0:
                        burn_in_phase_v[:] = False
                        log("Stopping burn in after %d iterations" % (iteration_num), INFO)


                    #max_beta = None
                    #if max_beta is not None:
                    #    threshold_ravel = max_beta * scale_factors_m.ravel()
                    #    if np.sum(sample_betas_m.ravel() > threshold_ravel) > 0:
                    #        log("Capped %d sample betas" % np.sum(sample_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        sample_betas_mask = sample_betas_m.ravel() > threshold_ravel
                    #        sample_betas_m.ravel()[sample_betas_mask] = threshold_ravel[sample_betas_mask]
                    #    if np.sum(curr_betas_m.ravel() > threshold_ravel) > 0:
                    #        log("Capped %d curr betas" % np.sum(curr_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        curr_betas_mask = curr_betas_m.ravel() > threshold_ravel
                    #        curr_betas_m.ravel()[curr_betas_mask] = threshold_ravel[curr_betas_mask]

                    return (sample_betas_m, sample_postp_m, curr_betas_m, curr_postp_m)

                #average over the posterior means instead of samples
                #these differ from sum_betas_v because those include the burn in phase
                avg_betas_m += np.sum(curr_post_means_t, axis=0)
                avg_betas2_m += np.sum(np.power(curr_post_means_t, 2), axis=0)
                avg_postp_m += np.sum(curr_postp_t, axis=0)
                num_avg += curr_post_means_t.shape[0]

                if iteration_num >= min_num_iter and num_avg > 1:
                    if gauss_seidel:
                        will_break = True
                    else:

                        #calculate these here for trace printing
                        avg_m = avg_betas_m
                        avg2_m = avg_betas2_m
                        sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
                        sem2_v = np.sum(sem2_m, axis=0)
                        zero_sem2_v = sem2_v == 0
                        sem2_v[zero_sem2_v] = 1
                        total_z_v = np.sqrt(np.sum(avg2_m / num_avg, axis=0) / sem2_v)
                        total_z_v[zero_sem2_v] = np.inf

                        log("Iteration %d: sum2=%.4g; sum sem2=%.4g; z=%.3g" % (iteration_num, np.sum(avg2_m / num_avg), np.sum(sem2_m), np.min(total_z_v)), TRACE)

                        min_z_sampling_var = 10
                        if np.all(total_z_v > min_z_sampling_var):
                            log("Desired precision achieved; stopping sampling")
                            will_break=True

                        #TODO: STILL FINALIZING HOW TO DO THIS
                        #avg_m = avg_betas_m
                        #avg2_m = avg_betas2_m

                        #sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
                        #zero_sem2_m = sem2_m == 0
                        #sem2_m[zero_sem2_m] = 1

                        #max_avg = np.max(np.abs(avg_m / num_avg))
                        #min_avg = np.min(np.abs(avg_m / num_avg))
                        #ref_val = max_avg - min_avg
                        #if ref_val == 0:
                        #    ref_val = np.sqrt(np.var(curr_post_means_t))
                        #    if ref_val == 0:
                        #        ref_val = 1

                        #max_sem = np.max(np.sqrt(sem2_m))
                        #max_percentage_error = max_sem / ref_val

                        #log("Iteration %d: ref_val=%.3g; max_sem=%.3g; max_ratio=%.3g" % (iteration_num, ref_val, max_sem, max_percentage_error))
                        #if max_percentage_error < max_frac_sem:
                        #    log("Desired precision achieved; stopping sampling")
                        #    break
                        
            else:
                if update_hyper_p or update_hyper_sigma:
                    h2 = 0
                    for i in range(num_parallel):
                        if use_X:
                            h2 += curr_betas_m[i,:].dot(curr_betas_m[i,:])
                        else:
                            if multiple_V:
                                cur_V = V[i,:,:]
                            else:
                                cur_V = V
                            if sparse_V:
                                h2 += V.dot(curr_betas_m[i,:].T).T.dot(curr_betas_m[i,:])
                            else:
                                h2 += curr_betas_m[i,:].dot(cur_V).dot(curr_betas_m[i,:])
                    h2 /= num_parallel

                    new_p = np.mean((np.sum(curr_betas_t > 0, axis=2) + num_p_pseudo) / float(curr_betas_t.shape[2] + num_p_pseudo))

                    if self.sigma_power is not None:
                        new_sigma2 = h2 / np.mean(np.sum(np.power(scale_factors_m, self.sigma_power), axis=1))
                    else:
                        new_sigma2 = h2 / num_gene_sets

                    if num_missing_gene_sets:
                        missing_scale_factor = num_gene_sets / (num_gene_sets + num_missing_gene_sets)
                        new_sigma2 *= missing_scale_factor
                        new_p *= missing_scale_factor

                    if p_noninf_inflate != 1:
                        log("Inflating p by %.3g" % p_noninf_inflate, DEBUG)
                        new_p *= p_noninf_inflate

                    if abs(new_sigma2 - self.sigma2) / self.sigma2 < eps and abs(new_p - self.p) / self.p < eps:
                        log("Sigma converged to %.4g; p converged to %.4g" % (self.sigma2, self.p), TRACE)
                        update_hyper_sigma = False
                        update_hyper_p = False
                    else:
                        if update_hyper_p:
                            log("Updating p from %.4g to %.4g" % (self.p, new_p), TRACE)
                            if not update_hyper_sigma and adjust_hyper_sigma_p:
                                #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
                                #if we are only updating p, and adjusting sigma, we will leave the conditional_sigma constant, which means scaling the sigma
                                new_sigma2 = self.sigma2 / self.p * new_p
                                log("Updating sigma from %.4g to %.4g to maintain constant sigma/p" % (self.sigma2, new_sigma2), TRACE)
                                #we need to adjust the total sigma to keep the conditional sigma constant
                                self.set_sigma(new_sigma2, self.sigma_power)
                            self.set_p(new_p)
                                
                        if update_hyper_sigma:
                            if not sigma_underflow:
                                log("Updating sigma from %.4g to %.4g ( sqrt(sigma2/p)=%.4g )" % (self.sigma2, new_sigma2, np.sqrt(new_sigma2 / self.p)), TRACE)

                            lower_bound = 2e-3

                            if sigma_underflow or new_sigma2 / self.p < lower_bound:
                                
                                #first, try the heuristic of setting sigma2 so that strongest gene set has maximum possible p_bar

                                max_e_beta2 = np.argmax(beta_tildes_m / ses_m)

                                max_se2 = se2s_m.ravel()[max_e_beta2]
                                max_beta_tilde = beta_tildes_m.ravel()[max_e_beta2]
                                max_beta_tilde2 = np.square(max_beta_tilde)

                                #OLD inference
                                #make sigma/p easily cover the observation
                                #new_sigma2 = (max_beta_tilde2 - max_se2) * self.p
                                #make sigma a little bit smaller so that the top gene set is a little more of an outlier
                                #new_sigma2 /= sigma_num_devs_to_top

                                #NEW inference
                                max_beta = np.sqrt(max_beta_tilde2 - max_se2)
                                correct_sigma2 = self.p * np.square(max_beta / np.abs(scipy.stats.norm.ppf(1 / float(curr_betas_t.shape[2]) * self.p * 2)))
                                new_sigma2 = correct_sigma2

                                if new_sigma2 / self.p <= lower_bound:
                                    new_sigma2_from_top = new_sigma2
                                    new_sigma2 = lower_bound * self.p
                                    log("Sigma underflow including with determination from top gene set (%.4g)! Setting sigma to lower bound (%.4g * %.4g = %.4g) and no updates" % (new_sigma2_from_top, lower_bound, self.p, new_sigma2), TRACE)
                                else:
                                    log("Sigma underflow! Setting sigma determined from top gene set (%.4g) and no updates" % new_sigma2, TRACE)

                                if self.sigma_power is not None:

                                    #gene set specific sigma is internal sigma2 multiplied by scale_factor ** power
                                    #new_sigma2 is final sigma
                                    #so store internal value as final divided by average power

                                    #use power learned from mouse
                                    #using average across gene sets makes it sensitive to distribution of gene sets
                                    #need better solution for learning; since we are hardcoding from top gene set, just use mouse value
                                    new_sigma2 = new_sigma2 / np.power(self.MEAN_MOUSE_SCALE, self.sigma_power)

                                    #if np.sum([~is_dense_gene_set_m]) > 0:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
                                    #else:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m.ravel()), self.sigma_power)

                                    #if is_dense_gene_set_m.ravel()[max_e_beta2]:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
                                    #else:
                                    #    new_sigma2 = new_sigma2 / np.power(scale_factors_m.ravel()[max_e_beta2], self.sigma_power)

                                if not update_hyper_p and adjust_hyper_sigma_p:
                                    #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
                                    #if we are only sigma p, and adjusting p, we will leave the conditional_sigma constant, which means scaling the p
                                    new_p = self.p / self.sigma2 * new_sigma2
                                    log("Updating p from %.4g to %.4g to maintain constant sigma/p" % (self.p, new_p), TRACE)
                                    #we need to adjust the total sigma to keep the conditional sigma constant
                                    self.set_p(new_p)

                                self.set_sigma(new_sigma2, self.sigma_power)
                                sigma_underflow = True

                                #update_hyper_sigma = False
                                #restarting sampling with sigma2 fixed to initial value due to underflow
                                #update_hyper_p = False

                                #reset loop state
                                #iteration_num = 0
                                #curr_post_means_t = np.zeros(tensor_shape)
                                #curr_postp_t = np.ones(tensor_shape)
                                #curr_betas_t = scipy.stats.norm.rvs(0, np.std(beta_tildes_m), tensor_shape)                            
                                #avg_betas_m = np.zeros(matrix_shape)
                                #avg_betas2_m = np.zeros(matrix_shape)
                                #avg_postp_m = np.zeros(matrix_shape)
                                #num_avg = 0
                                #sum_betas_t = np.zeros(tensor_shape)
                                #sum_betas2_t = np.zeros(tensor_shape)
                            else:
                                self.set_sigma(new_sigma2, self.sigma_power)

                            #update the matrix forms of these variables
                            orig_sigma2_m *= new_sigma2 / np.mean(orig_sigma2_m)
                            if self.sigma_power is not None:
                                #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
                                sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

                                #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
                                if np.sum(is_dense_gene_set_m) > 0:
                                    if np.sum(~is_dense_gene_set_m) > 0:
                                        #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                                    else:
                                        #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                            else:
                                sigma2_m = orig_sigma2_m

                            ps_m *= new_p / np.mean(ps_m)

            if betas_trace_out is not None:
                for parallel_num in range(num_parallel):
                    for chain_num in range(num_chains):
                        for i in range(num_gene_sets):
                            gene_set = i
                            if betas_trace_gene_sets is not None and len(betas_trace_gene_sets) == num_gene_sets:
                                gene_set = betas_trace_gene_sets[i]

                            betas_trace_fh.write("%d\t%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num, parallel_num+1, chain_num+1, gene_set, curr_post_means_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_postp_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], beta_tildes_m[parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i], beta_tildes_m[parallel_num,i], ses_m[parallel_num,i], sigma2_m[parallel_num,i] if len(np.shape(sigma2_m)) > 0 else sigma2_m, ps_m[parallel_num,i] if len(np.shape(ps_m)) > 0 else ps_m, R_m[parallel_num,i], R_m[parallel_num,i] * beta_weights_m[parallel_num,i], sem2_m[parallel_num, i]))

                betas_trace_fh.flush()

            if will_break:
                break


        if betas_trace_out is not None:
            betas_trace_fh.close()

            #log("%d\t%s" % (iteration_num, "\t".join(["%.3g\t%.3g" % (curr_betas_m[i,0], (np.mean(sum_betas_m, axis=0) / iteration_num)[i]) for i in range(curr_betas_m.shape[0])])), TRACE)

        avg_betas_m /= num_avg
        avg_postp_m /= num_avg

        if num_parallel == 1:
            avg_betas_m = avg_betas_m.flatten()
            avg_postp_m = avg_postp_m.flatten()

        #max_beta = None
        #if max_beta is not None:
        #    threshold_ravel = max_beta * scale_factors_m.ravel()
        #    if np.sum(avg_betas_m.ravel() > threshold_ravel) > 0:
        #        log("Capped %d sample betas" % np.sum(avg_betas_m.ravel() > threshold_ravel), DEBUG)
        #        avg_betas_mask = avg_betas_m.ravel() > threshold_ravel
        #        avg_betas_m.ravel()[avg_betas_mask] = threshold_ravel[avg_betas_mask]

        frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
        if frac_increase > 0.01:
            warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
            printed_warning_increase = True

        frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
        if frac_opposite > 0.01:
            warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
            printed_warning_swing = False

        return (avg_betas_m, avg_postp_m)


    #store Y value
    #Y is whitened if Y_corr_m is not null
    def _set_Y(self, Y, Y_for_regression=None, Y_exomes=None, Y_positive_controls=None, Y_corr_m=None, store_cholesky=True, store_corr_sparse=False, skip_V=False, skip_scale_factors=False, min_correlation=0):
        log("Setting Y", TRACE)

        self.last_X_block = None
        if Y_corr_m is not None:

            if min_correlation is not None:
                #set things too far away to 0
                Y_corr_m[Y_corr_m <= 0] = 0

            #remove bands at the end that are all zero
            keep_mask = np.array([True] * len(Y_corr_m))
            for i in range(len(Y_corr_m)-1, -1, -1):
                if sum(Y_corr_m[i] != 0) == 0:
                    keep_mask[i] = False
                else:
                    break
            if sum(keep_mask) > 0:
                Y_corr_m = Y_corr_m[keep_mask]

            #scale factor for diagonal to ensure non-singularity
            self.y_corr = copy.copy(Y_corr_m)

            y_corr_diags = [self.y_corr[i,:(len(self.y_corr[i,:]) - i)] for i in range(len(self.y_corr))]
            y_corr_sparse = sparse.csc_matrix(sparse.diags(y_corr_diags + y_corr_diags[1:], list(range(len(y_corr_diags))) + list(range(-1, -len(y_corr_diags), -1))))            

            if store_cholesky:
                self.y_corr_cholesky = self._get_y_corr_cholesky(Y_corr_m)
                log("Banded cholesky matrix: shape %s, %s" % (self.y_corr_cholesky.shape[0], self.y_corr_cholesky.shape[1]), DEBUG)
                #whitened
                self.Y_w = scipy.linalg.solve_banded((self.y_corr_cholesky.shape[0]-1, 0), self.y_corr_cholesky, Y)
                na_mask = ~np.isnan(self.Y_w)
                self.y_w_var = np.var(self.Y_w[na_mask])
                self.y_w_mean = np.mean(self.Y_w[na_mask])
                self.Y_w = self.Y_w - self.y_w_mean
                #fully whitened
                self.Y_fw = scipy.linalg.cho_solve_banded((self.y_corr_cholesky, True), Y)
                na_mask = ~np.isnan(self.Y_fw)
                self.y_fw_var = np.var(self.Y_fw[na_mask])
                self.y_fw_mean = np.mean(self.Y_fw[na_mask])
                self.Y_fw = self.Y_fw - self.y_fw_mean

                #update the scale factors and mean shifts for the whitened X
                self._set_X(self.X_orig, self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)
                if self.X_orig_missing_gene_sets is not None and not skip_scale_factors:
            
                    (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

            if store_corr_sparse:

                self.y_corr_sparse = y_corr_sparse

        if Y is not None:
            na_mask = ~np.isnan(Y)
            self.y_var = np.var(Y[na_mask])
        else:
            self.y_var = None
        #DO WE NEED THIS???
        #self.y_mean = np.mean(Y[na_mask])
        #self.Y = Y - self.y_mean
        self.Y = Y
        self.Y_for_regression = Y_for_regression
        self.Y_exomes = Y_exomes
        self.Y_positive_controls = Y_positive_controls

    def _get_y_corr_cholesky(self, Y_corr_m):
        Y_corr_m_copy = copy.copy(Y_corr_m)
        diag_add = 0.05
        while True:
            try:
                Y_corr_m_copy[0,:] += diag_add
                Y_corr_m_copy /= (1 + diag_add)
                y_corr_cholesky = scipy.linalg.cholesky_banded(Y_corr_m_copy, lower=True)
                return y_corr_cholesky
            except np.linalg.LinAlgError:
                pass

    def _whiten(self, matrix, corr_cholesky, whiten=True, full_whiten=False):
        if full_whiten:
            #fully whiten, by sigma^{-1}; useful for optimization
            matrix = scipy.linalg.cho_solve_banded((corr_cholesky, True), matrix, overwrite_b=True)
        elif whiten:
            #whiten X_b by sigma^{-1/2}
            matrix = scipy.linalg.solve_banded((corr_cholesky.shape[0]-1, 0), corr_cholesky, matrix, overwrite_ab=True)
        return matrix

    #return an iterator over chunks of X in dense format
    #useful when want to conduct matrix calculations for which dense arrays are much faster, but don't have enough memory to cast all of X to dense
    #full_whiten (which multiplies by C^{-1} takes precedence over whiten, which multiplies by C^{1/2}, but whiten defaults to true
    #if mean_shifts/scale_factors are passed in, then shift/rescale the blocks. This is done *before* any whitening
    def _get_X_blocks(self, whiten=True, full_whiten=False, get_missing=False, start_batch=0, mean_shifts=None, scale_factors=None):
        X_orig = self.X_orig
        if get_missing:
            X_orig = self.X_orig_missing_gene_sets

        for (X_b, begin, end, batch) in self._get_X_blocks_internal(X_orig, self.y_corr_cholesky, whiten=whiten, full_whiten=full_whiten, start_batch=start_batch, mean_shifts=mean_shifts, scale_factors=scale_factors):
            yield (X_b, begin, end, batch)

    def _get_num_X_blocks(self, X_orig, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return int(np.ceil(X_orig.shape[1] / batch_size))

    def _get_X_size_mb(self, X_orig=None):
        if X_orig is None:
            X_orig = self.X_orig
        return (self.X_orig.data.nbytes + self.X_orig.indptr.nbytes + self.X_orig.indices.nbytes) / 1024 / 1024

    def _get_X_blocks_internal(self, X_orig, y_corr_cholesky, whiten=True, full_whiten=False, start_batch=0, mean_shifts=None, scale_factors=None):

        if y_corr_cholesky is None:
            #explicitly turn these off to help with caching
            whiten = False
            full_whiten = False

        num_batches = self._get_num_X_blocks(X_orig)

        consider_cache = X_orig is self.X_orig and num_batches == 1 and mean_shifts is None and scale_factors is None

        for batch in range(start_batch, num_batches):
            log("Getting X%s block batch %s (%s)" % ("_missing" if X_orig is self.X_orig_missing_gene_sets else "", batch, "fully whitened" if full_whiten else ("whitened" if whiten else "original")), TRACE)
            begin = batch * self.batch_size
            end = (batch + 1) * self.batch_size
            if end > X_orig.shape[1]:
                end = X_orig.shape[1]

            if self.last_X_block is not None and consider_cache and self.last_X_block[1:] == (whiten, full_whiten, begin, end, batch):
                log("Using cache!", TRACE)
                yield (self.last_X_block[0], begin, end, batch)
            else:
                X_b = X_orig[:,begin:end].toarray()
                if mean_shifts is not None:
                    X_b = X_b - mean_shifts[begin:end]
                if scale_factors is not None:
                    X_b = X_b / scale_factors[begin:end]

                if whiten or full_whiten:
                    X_b = self._whiten(X_b, y_corr_cholesky, whiten=whiten, full_whiten=full_whiten)

                #only cache if we are accessing the original X
                if consider_cache:
                    self.last_X_block = (X_b, whiten, full_whiten, begin, end, batch)
                else:
                    self.last_X_block = None

                yield (X_b, begin, end, batch)

    def _get_fraction_non_missing(self):
        if self.gene_sets_missing is not None and self.gene_sets is not None:
            fraction_non_missing = float(len(self.gene_sets)) / float(len(self.gene_sets_missing) + len(self.gene_sets))        
        else:
            fraction_non_missing = 1
        return fraction_non_missing
    
    def _calc_X_shift_scale(self, X, y_corr_cholesky=None):
        if y_corr_cholesky is None:
            if sparse.issparse(X):
                mean_shifts = X.sum(axis=0).A1 / X.shape[0]
                scale_factors = np.sqrt(X.power(2).sum(axis=0).A1 / X.shape[0] - np.square(mean_shifts))
            else:
                mean_shifts = np.mean(X, axis=0)
                scale_factors = np.std(X, axis=0)
        else:
            scale_factors = np.array([])
            mean_shifts = np.array([])
            for X_b, begin, end, batch in self._get_X_blocks_internal(X, y_corr_cholesky):
                (cur_mean_shifts, cur_scale_factors) = self._calc_shift_scale(X_b)
                mean_shifts = np.append(mean_shifts, cur_mean_shifts)
                scale_factors = np.append(scale_factors, cur_scale_factors)
        return (mean_shifts, scale_factors)

    def _calc_shift_scale(self, X_b):
        mean_shifts = []
        scale_factors = []
        for i in range(X_b.shape[1]):
            X_i = X_b[:,i]
            mean_shifts.append(np.mean(X_i))
            scale_factor = np.std(X_i)
            if scale_factor == 0:
                scale_factor = 1
            scale_factors.append(scale_factor)
        return (np.array(mean_shifts), np.array(scale_factors))

    #store a (possibly unnormalized) X matrix
    #the passed in X should be a sparse matrix, with 0/1 values
    #does normalization
    def _set_X(self, X_orig, genes, gene_sets, skip_V=False, skip_scale_factors=False, skip_N=True):

        log("Setting X", TRACE)

        if X_orig is not None:
            if not len(genes) == X_orig.shape[0]:
                bail("Dimension mismatch when setting X: %d genes but %d rows in X" % (len(genes), X_orig.shape[0]))
            if not len(gene_sets) == X_orig.shape[1]:
                bail("Dimension mismatch when setting X: %d gene sets but %d columns in X" % (len(gene_sets), X_orig.shape[1]))

        if self.X_orig is not None and X_orig is self.X_orig and genes is self.genes and gene_sets is self.gene_sets and ((self.y_corr_cholesky is None and not self.scale_is_for_whitened) or (self.y_corr_cholesky is not None and self.scale_is_for_whitened)):
            return
        
        self.last_X_block = None

        self.genes = genes

        if self.genes is not None:
            self.gene_to_ind = self._construct_map_to_ind(self.genes)
        else:
            self.gene_to_ind = None

        self.gene_sets = gene_sets
        if self.gene_sets is not None:
            self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)
        else:
            self.gene_set_to_ind = None

        self.X_orig = X_orig
        if self.X_orig is not None:
            self.X_orig.eliminate_zeros()

        if self.X_orig is None:
            self.X_orig_missing_genes = None
            self.X_orig_missing_genes_missing_gene_sets = None
            self.X_orig_missing_gene_sets = None
            self.last_X_block = None
            return

        if not skip_N:
            self.gene_N = self.get_col_sums(self.X_orig, axis=1)

        #self.X = self.X_orig.todense().astype(float)
        if not skip_scale_factors:
            self._set_scale_factors()

        #X = self.X_orig.todense().astype(float)
        #for i in range(self.X_orig.shape[1]):
        #    X[:,i] = (X[:,i] - self.mean_shifts[i]) / self.scale_factors[i]
        #self.V = X.T.dot(X) / len(self.genes)

    def _set_scale_factors(self):

        log("Calculating scale factors and mean shifts", TRACE)
        (self.mean_shifts, self.scale_factors) = self._calc_X_shift_scale(self.X_orig, self.y_corr_cholesky)

        #flag to indicate whether these scale factors correspond to X_orig or the (implicit) whitened version
        if self.y_corr_cholesky is not None:
            self.scale_is_for_whitened = True
        else:
            self.scale_is_for_whitened = False

    def _get_V(self):
        if self.X_orig is not None:
            log("Calculating internal V", TRACE)
            return self._calculate_V()
        else:
            return None

    def _calculate_V(self, X_orig=None, y_corr_cholesky=None, mean_shifts=None, scale_factors=None):
        if X_orig is None:
            X_orig = self.X_orig
        if mean_shifts is None:
            mean_shifts = self.mean_shifts
        if scale_factors is None:
            scale_factors = self.scale_factors
        if y_corr_cholesky is None:
            y_corr_cholesky = self.y_corr_cholesky
        return self._calculate_V_internal(X_orig, y_corr_cholesky, mean_shifts, scale_factors)

    def _calculate_V_internal(self, X_orig, y_corr_cholesky, mean_shifts, scale_factors, y_corr_sparse=None):
        log("Calculating V for X with dimensions %d x %d" % (X_orig.shape[0], X_orig.shape[1]), TRACE)

        #TEMP
        if y_corr_cholesky is not None:

            if self._get_num_X_blocks(X_orig) == 1:
                whiten1 = True
                full_whiten1 = False
                whiten2 = True
                full_whiten2 = False
            else:
                whiten1 = False
                full_whiten1 = True
                whiten2 = False
                full_whiten2 = False

            V = None
            if X_orig is not None:
                #X_b is whitened

                for X_b1, begin1, end1, batch1 in self._get_X_blocks_internal(X_orig, y_corr_cholesky, whiten=whiten1, full_whiten=full_whiten1, mean_shifts=mean_shifts, scale_factors=scale_factors):
                    cur_V = None

                    if y_corr_sparse is not None:
                        X_b1 = y_corr_sparse.dot(X_b1)

                    for X_b2, begin2, end2, batch2 in self._get_X_blocks_internal(X_orig, y_corr_cholesky, whiten=whiten2, full_whiten=full_whiten2, mean_shifts=mean_shifts, scale_factors=scale_factors):
                        V_block = self._compute_V(X_b1, 0, 1, X_orig2=X_b2, mean_shifts2=0, scale_factors2=1)
                        if cur_V is None:
                            cur_V = V_block
                        else:
                            cur_V = np.hstack((cur_V, V_block))
                    if V is None:
                        V = cur_V
                    else:
                        V = np.vstack((V, cur_V))
        else:
            V = self._compute_V(X_orig, mean_shifts, scale_factors)

        return V

    #calculate V between X_orig and X_orig2
    #X_orig2 can be dense or sparse, but if it is sparse than X_orig must also be sparse
    def _compute_V(self, X_orig, mean_shifts, scale_factors, rows = None, X_orig2 = None, mean_shifts2 = None, scale_factors2 = None):
        if X_orig2 is None:
            X_orig2 = X_orig
        if mean_shifts2 is None:
            mean_shifts2 = mean_shifts
        if scale_factors2 is None:
            scale_factors2 = scale_factors
        if rows is None:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                dot_product = X_orig.T.dot(X_orig2)                
            else:
                dot_product = X_orig.T.dot(X_orig2).toarray().astype(float)
        else:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                dot_product = X_orig[:,rows].T.dot(X_orig2)
            else:
                dot_product = X_orig[:,rows].T.dot(X_orig2).toarray().astype(float)
            mean_shifts = mean_shifts[rows]
            scale_factors = scale_factors[rows]

        return (dot_product/X_orig.shape[0] - np.outer(mean_shifts, mean_shifts2)) / (np.outer(scale_factors, scale_factors2) + 1e-10)
            
    #by default, find batches of uncorrelated genes (for use in gibbs)
    #option to find batches of correlated (pass in batch size)
    #this is a greedy addition method
    #if have sort_values, will greedily add from lowest value to higher value
    def _compute_gene_set_batches(self, V=None, X_orig=None, mean_shifts=None, scale_factors=None, use_sum=True, max_allowed_batch_correlation=None, find_correlated_instead=None, sort_values=None, stop_at=None, tag="gene sets"):
        gene_set_masks = []

        if max_allowed_batch_correlation is None:
            if use_sum:
                max_allowed_batch_correlation = 0.5
            else:
                max_allowed_batch_correlation = 0.1

        if find_correlated_instead is not None:
            if find_correlated_instead < 1:
                bail("Need batch size of at least 1")

        if use_sum:
            combo_fn = np.sum
        else:
            combo_fn = np.max

        use_X = False
        if V is not None and len(V.shape) == 3:
            num_gene_sets = V.shape[1]
            not_included_gene_sets = np.full((V.shape[0], num_gene_sets), True)
        elif V is not None:
            num_gene_sets = V.shape[0]
            not_included_gene_sets = np.full(num_gene_sets, True)
        else:
            assert(mean_shifts.shape == scale_factors.shape)
            if len(mean_shifts.shape) > 1:
                if mean_shifts.shape[0] == 1:
                    mean_shifts = np.squeeze(mean_shifts, axis=0)
                    scale_factors = np.squeeze(scale_factors, axis=0)
                elif np.all(np.isclose(np.var(mean_shifts, axis=0), 0)):
                    mean_shifts = np.mean(mean_shifts, axis=0)
                    scale_factors = np.mean(scale_factors, axis=0)
                else:
                    bail("Error: can't have different mean shifts across chains")
            if X_orig is None or mean_shifts is None or scale_factors is None:
                bail("Need X_orig or V for this operation")
            num_gene_sets = X_orig.shape[1]
            not_included_gene_sets = np.full(num_gene_sets, True)
            use_X = True

        log("Batching %d %s..." % (num_gene_sets, tag), INFO)
        if use_X:
            log("Using low memory mode", DEBUG)

        indices = np.array(range(num_gene_sets))

        if sort_values is None:
            sort_values = indices

        total_added = 0

        while np.any(not_included_gene_sets):
            if V is not None and len(V.shape) == 3:
                #batches if multiple_V

                current_mask = np.full((V.shape[0], num_gene_sets), False)
                #set the first gene set in each row to True
                for c in range(V.shape[0]):

                    sorted_remaining_indices = sorted(indices[not_included_gene_sets[c,:]], key=lambda k: sort_values[k])
                    #seed with the first gene not already included
                    if len(sorted_remaining_indices) == 0:
                        continue

                    first_gene_set = sorted_remaining_indices[0]
                    current_mask[c,first_gene_set] = True
                    not_included_gene_sets[c,first_gene_set] = False
                    sorted_remaining_indices = sorted_remaining_indices[1:]

                    if find_correlated_instead:
                        #WARNING: THIS HAS NOT BEEN TESTED
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets[c,:])[0]
                        ordered_indices = index_map[np.argsort(-V[c,first_gene_set,:])[not_included_gene_sets[c,:]]]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[c,indices_to_add] = True
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[c,i,current_mask[c,:]]) < max_allowed_batch_correlation:
                                current_mask[c,i] = True
                                not_included_gene_sets[c,i] = False
            else:
                sorted_remaining_indices = sorted(indices[not_included_gene_sets], key=lambda k: sort_values[k])
                #batches if one V
                current_mask = np.full(num_gene_sets, False)
                #seed with the first gene not already included
                first_gene_set = sorted_remaining_indices[0]
                current_mask[first_gene_set] = True
                not_included_gene_sets[first_gene_set] = False
                sorted_remaining_indices = sorted_remaining_indices[1:]

                if V is not None:
                    if find_correlated_instead:
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-V[first_gene_set,not_included_gene_sets])]
                        #map these to the original ones
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[i,current_mask]) < max_allowed_batch_correlation:
                                current_mask[i] = True
                                not_included_gene_sets[i] = False
                else:
                    assert(scale_factors.shape == mean_shifts.shape)

                    if find_correlated_instead:
                        cur_V = self._compute_V(X_orig[:,first_gene_set], mean_shifts[first_gene_set], scale_factors[first_gene_set], X_orig2=X_orig[:,not_included_gene_sets], mean_shifts2=mean_shifts[not_included_gene_sets], scale_factors2=scale_factors[not_included_gene_sets])
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-cur_V[0,:])]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        #cap out at batch_size gene sets to avoid memory of making whole V; this may reduce the batch size relative to optimal
                        #also, only add those not in mask already (since we are searching only these in V)
                        max_to_add = self.batch_size
                        V_to_generate_mask = copy.copy(not_included_gene_sets)
                        if np.sum(V_to_generate_mask) > max_to_add:
                            assert(len(sorted_remaining_indices) == np.sum(not_included_gene_sets))
                            V_to_generate_mask[sort_values > sort_values[sorted_remaining_indices[max_to_add]]] = False

                        V_to_generate_mask[first_gene_set] = True
                        cur_V = self._compute_V(X_orig[:,V_to_generate_mask], mean_shifts[V_to_generate_mask], scale_factors[V_to_generate_mask])
                        indices_not_included = indices[V_to_generate_mask]
                        sorted_cur_V_indices = sorted(range(cur_V.shape[0]), key=lambda k: sort_values[indices_not_included[k]])
                        for i in sorted_cur_V_indices:
                            if combo_fn(cur_V[i,current_mask[V_to_generate_mask]]) < max_allowed_batch_correlation:
                                current_mask[indices_not_included[i]] = True
                                not_included_gene_sets[indices_not_included[i]] = False

            gene_set_masks.append(current_mask)
            #log("Batch %d; %d gene sets" % (len(gene_set_masks), sum(current_mask)), TRACE)
            total_added += np.sum(current_mask)
            if stop_at is not None and total_added >= stop_at:
                log("Breaking at %d" % total_added, TRACE)
                break

        denom = 1
        if V is not None and len(V.shape) == 3:
            denom = V.shape[0]

        sizes = [float(np.sum(x)) / denom for x in gene_set_masks]
        log("Batched %d %s into %d batches; size range %d - %d" % (num_gene_sets, tag, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes)  if len(sizes) > 0 else 0), DEBUG)

        return gene_set_masks

    #sort the genes in the matrices
    #does not alter genes already subseet
    def _sort_genes(self, sorted_gene_indices, skip_V=False, skip_scale_factors=False):

        log("Sorting genes", TRACE)
        if self.y_corr_cholesky is not None:
            #FIXME: subset the cholesky matrix here
            bail("Sorting genes after setting correlation matrix is not yet implemented")

        self.genes = [self.genes[i] for i in sorted_gene_indices]
        self.gene_to_ind = self._construct_map_to_ind(self.genes)

        index_map = {sorted_gene_indices[i]: i for i in range(len(sorted_gene_indices))}

        if self.X_orig is not None:
            #reset the X matrix and scale factors
            self._set_X(sparse.csc_matrix((self.X_orig.data, [index_map[x] for x in self.X_orig.indices], self.X_orig.indptr), shape=self.X_orig.shape), self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)

        if self.X_orig_missing_gene_sets is not None:
            #if we've already removed gene sets, then we need remove the genes from them too
            
            self.X_orig_missing_gene_sets = sparse.csc_matrix((self.X_orig_missing_gene_sets.data, [index_map[x] for x in self.X_orig_missing_gene_sets.indices], self.X_orig_missing_gene_sets.indptr), shape=self.X_orig_missing_gene_sets.shape)
            #need to recompute these
            (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

        if self.huge_signal_bfs is not None:
            #reset the X matrix and scale factors
            self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, [index_map[x] for x in self.huge_signal_bfs.indices], self.huge_signal_bfs.indptr), shape=self.huge_signal_bfs.shape)

        if self.huge_signal_bfs_for_regression is not None:
            #reset the X matrix and scale factors
            self.huge_signal_bfs_for_regression = sparse.csc_matrix((self.huge_signal_bfs_for_regression.data, [index_map[x] for x in self.huge_signal_bfs_for_regression.indices], self.huge_signal_bfs_for_regression.indptr), shape=self.huge_signal_bfs_for_regression.shape)

        index_map_rev = {i: sorted_gene_indices[i] for i in range(len(sorted_gene_indices))}

        if self.gene_covariates is not None:
            self.gene_covariates = self.gene_covariates[[index_map_rev[x] for x in range(self.gene_covariates.shape[0])],:]
            self.gene_covariate_zs = self.gene_covariate_zs[[index_map_rev[x] for x in range(self.gene_covariate_zs.shape[0])],:]

        if self.gene_covariate_adjustments is not None:
            self.gene_covariate_adjustments = self.gene_covariate_adjustments[[index_map_rev[x] for x in range(self.gene_covariate_adjustments.shape[0])]]

        if self.gene_covariates_mask is not None:
            self.gene_covariates_mask = self.gene_covariates_mask[[index_map_rev[x] for x in range(self.gene_covariates_mask.shape[0])]]

        if self.gene_pheno_combined_prior_Ys is not None or self.gene_pheno_Y is not None or self.gene_pheno_priors is not None:
            if self.gene_pheno_combined_prior_Ys is not None:
                self.gene_pheno_combined_prior_Ys = self.gene_pheno_combined_prior_Ys[[index_map_rev[x] for x in range(self.gene_pheno_combined_prior_Ys.shape[0])],:]
            if self.gene_pheno_Y is not None:
                self.gene_pheno_Y = self.gene_pheno_Y[[index_map_rev[x] for x in range(self.gene_pheno_Y.shape[0])],:]
            if self.gene_pheno_priors is not None:
                self.gene_pheno_priors = self.gene_pheno_priors[[index_map_rev[x] for x in range(self.gene_pheno_priors.shape[0])],:]

        if self.gene_factor_gene_mask is not None:
            self.gene_factor_gene_mask = self.gene_factor_gene_mask[[index_map_rev[x] for x in range(self.gene_pheno_combined_prior_Ys.shape[0])]]

        if self.gene_prob_factor_vector is not None:
            self.gene_prob_factor_vector = self.gene_prob_factor_vector[[index_map_rev[x] for x in range(self.gene_prob_factor_vector.shape[0])]]

        if self.exp_gene_factors is not None:
            self.exp_gene_factors = self.exp_gene_factors[[index_map_rev[x] for x in range(self.exp_gene_factors.shape[0])],:]

            
        self.exp_gene_factors = None #anchor-agnostic factor loadings
        self.gene_prob_factor_vector = None #outer product of this with factor loadings gives anchor specific loadings


        if self.gene_N is not None:
            self.gene_N = self.gene_N[sorted_gene_indices]
        if self.gene_ignored_N is not None:
            self.gene_ignored_N = self.gene_ignored_N[sorted_gene_indices]

        for x in [self.Y, self.Y_for_regression, self.Y_uncorrected, self.Y_exomes, self.Y_positive_controls, self.Y_w, self.Y_fw, self.priors, self.priors_adj, self.combined_prior_Ys, self.combined_prior_Ys_adj, self.combined_Ds, self.Y_orig, self.Y_for_regression_orig, self.Y_w_orig, self.Y_fw_orig, self.priors_orig, self.priors_adj_orig]:
            if x is not None:
                x[:] = np.array([x[i] for i in sorted_gene_indices])

    def _prune_gene_sets(self, prune_value, prune_deterministically=False, max_size=5000, keep_missing=False, ignore_missing=False, skip_V=False, X_orig=None, gene_sets=None, rank_vector=None, do_internal_pruning=True):

        if X_orig is None:
            X_orig = self.X_orig
            mean_shifts = self.mean_shifts
            scale_factors = self.scale_factors
        else:
            (mean_shifts, scale_factors) = self._calc_X_shift_scale(X_orig)
        name = ""
        if gene_sets is None:
            gene_sets = self.gene_sets
            name = " gene sets"
        if rank_vector is None:
            rank_vector = self.p_values

        if gene_sets is None or len(gene_sets) == 0:
            return
        if X_orig is None:
            return
        if prune_value > 1:
            return

        log("Pruning%s at %.3g..." % (name, prune_value), DEBUG)

        keep_mask = np.array([False] * len(gene_sets))
        remove_gene_sets = set()

        #keep total to batch_size ** 2

        batch_size = int(max_size ** 2 / X_orig.shape[1])
        num_batches = int(X_orig.shape[1] / batch_size) + 1

        for batch in range(num_batches):
            begin = batch * batch_size
            end = (batch + 1) * batch_size
            if end > X_orig.shape[1]:
                end = X_orig.shape[1]

            X_b1  = X_orig[:,begin:end]

            V_block = self._compute_V(X_orig[:,begin:end], mean_shifts[begin:end], scale_factors[begin:end], X_orig2=X_orig, mean_shifts2=mean_shifts, scale_factors2=scale_factors)

            if rank_vector is not None and False and not prune_deterministically:
                gene_set_key = lambda i: rank_vector[i]
            else:
                gene_set_key = lambda i: np.abs(X_b1[:,i]).sum(axis=0)

            for gene_set_ind in sorted(range(len(gene_sets[begin:end])), key=gene_set_key):
                absolute_ind = gene_set_ind + begin
                if absolute_ind in remove_gene_sets:
                    continue
                keep_mask[absolute_ind] = True
                remove_gene_sets.update(np.where(np.abs(V_block[gene_set_ind,:]) > prune_value)[0])
        if np.sum(~keep_mask) > 0:
            if X_orig is self.X_orig and do_internal_pruning:
                self.subset_gene_sets(keep_mask, keep_missing=keep_missing, ignore_missing=ignore_missing, skip_V=skip_V)
                log("Pruning at %.3g resulted in %d%s (of original %d)" % (prune_value, len(gene_sets), name, len(keep_mask)))

        return keep_mask

    def _subset_genes(self, gene_mask, skip_V=False, overwrite_missing=False, skip_scale_factors=False, skip_Y=False):

        if not overwrite_missing and sum(np.logical_not(gene_mask)) == 0:
            return
       
        log("Subsetting genes", TRACE)

        if overwrite_missing:
            self.genes_missing = None
            self.priors_missing = None
            self.gene_N_missing = None
            self.gene_ignored_N_missing = None
            self.X_orig_missing_genes = None
            self.X_orig_missing_genes_missing_gene_sets = None

        self.genes_missing = (self.genes_missing if self.genes_missing is not None else []) + [self.genes[i] for i in range(len(self.genes)) if not gene_mask[i]]

        self.gene_missing_to_ind = self._construct_map_to_ind(self.genes_missing)
        
        self.genes = [self.genes[i] for i in range(len(self.genes)) if gene_mask[i]]
        self.gene_to_ind = self._construct_map_to_ind(self.genes)

        remove_mask = np.logical_not(gene_mask)

        if self.gene_N is not None:
            self.gene_N_missing = np.concatenate((self.gene_N_missing if self.gene_N_missing is not None else np.array([]), self.gene_N[remove_mask]))
        if self.gene_ignored_N is not None:
            self.gene_ignored_N_missing = np.concatenate((self.gene_ignored_N_missing if self.gene_ignored_N_missing is not None else np.array([]), self.gene_ignored_N[remove_mask]))

        if self.X_orig is not None:
            #store the genes that were removed for later
            X_orig_missing_genes = self.X_orig[remove_mask,:]
            if self.X_orig_missing_genes is not None:
                self.X_orig_missing_genes = sparse.csc_matrix(sparse.vstack([self.X_orig_missing_genes, X_orig_missing_genes]))
            else:
                self.X_orig_missing_genes = X_orig_missing_genes

            #reset the X matrix and scale factors
            self._set_X(self.X_orig[gene_mask,:], self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)
            zero = self.X_orig.sum(axis=0).A1

        if self.X_orig_missing_gene_sets is not None:

            X_orig_missing_genes_missing_gene_sets = self.X_orig_missing_gene_sets[remove_mask,:]
            if self.X_orig_missing_genes_missing_gene_sets is not None:
                self.X_orig_missing_genes_missing_gene_sets = sparse.csc_matrix(sparse.vstack([self.X_orig_missing_genes_missing_gene_sets, X_orig_missing_genes_missing_gene_sets]))
            else:
                self.X_orig_missing_genes_missing_gene_sets = X_orig_missing_genes_missing_gene_sets

            #if we've already removed gene sets, then we need remove the genes from them too
            self.X_orig_missing_gene_sets = self.X_orig_missing_gene_sets[gene_mask,:]
            #need to recompute these
            (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

        if self.gene_N is not None:
            self.gene_N = self.gene_N[gene_mask]
        if self.gene_ignored_N is not None:
            self.gene_ignored_N = self.gene_ignored_N[gene_mask]


        if self.exp_gene_factors is not None:
            self.exp_gene_factors = self.exp_gene_factors[gene_mask,:]
        if self.gene_factor_gene_mask is not None:
            self.gene_factor_gene_mask = self.gene_factor_gene_mask[gene_mask,:]

        if not skip_Y:
            if self.Y is not None:
                self._set_Y(self.Y[gene_mask], self.Y_for_regression[gene_mask] if self.Y_for_regression is not None else None, self.Y_exomes[gene_mask] if self.Y_exomes is not None else None, self.Y_positive_controls[gene_mask] if self.Y_positive_controls is not None else None, Y_corr_m=self.y_corr[:,gene_mask] if self.y_corr is not None else None, store_cholesky=self.y_corr_cholesky is not None, store_corr_sparse=self.y_corr_sparse is not None, skip_V=skip_V)

            if self.Y_uncorrected is not None:
                self.Y_uncorrected = self.Y_uncorrected[gene_mask]

            if self.huge_signal_bfs is not None:
                self.huge_signal_bfs = self.huge_signal_bfs[gene_mask,:]
            if self.huge_signal_bfs_for_regression is not None:
                self.huge_signal_bfs_for_regression = self.huge_signal_bfs_for_regression[gene_mask,:]

            if self.gene_covariates is not None:
                self.gene_covariates = self.gene_covariates[gene_mask,:]
            if self.gene_covariate_zs is not None:
                self.gene_covariate_zs = self.gene_covariate_zs[gene_mask,:]
            if self.gene_covariate_adjustments is not None:
                self.gene_covariate_adjustments = self.gene_covariate_adjustments[gene_mask]
            if self.gene_covariates_mask is not None:
                self.gene_covariates_mask = self.gene_covariates_mask[gene_mask]


            if self.priors is not None:
                self.priors_missing = (self.priors_missing if self.priors_missing is not None else []) + [self.priors[i] for i in range(len(self.priors)) if not gene_mask[i]]
                self.priors = self.priors[gene_mask]
            
            if self.priors_adj is not None:
                self.priors_adj = self.priors_adj[gene_mask]
            if self.combined_prior_Ys is not None:
                self.combined_prior_Ys = self.combined_prior_Ys[gene_mask]
            if self.combined_prior_Ys_adj is not None:
                self.combined_prior_Ys_adj = self.combined_prior_Ys_adj[gene_mask]
            if self.combined_Ds is not None:
                self.combined_Ds = self.combined_Ds[gene_mask]
            if self.Y_orig is not None:
                self.Y_orig = self.Y_orig[gene_mask]
            if self.Y_for_regression_orig is not None:
                self.Y_for_regression_orig = self.Y_for_regression_orig[gene_mask]
            if self.Y_w_orig is not None:
                self.Y_w_orig = self.Y_w_orig[gene_mask]
            if self.Y_fw_orig is not None:
                self.Y_fw_orig = self.Y_fw_orig[gene_mask]
            if self.priors_orig is not None:
                self.priors_missing_orig = (self.priors_missing_orig if self.priors_missing_orig is not None else []) + [self.priors_orig[i] for i in range(len(self.priors_orig)) if not gene_mask[i]]
                self.priors_orig = self.priors_orig[gene_mask]
            if self.priors_adj_orig is not None:
                self.priors_adj_missing_orig = (self.priors_adj_missing_orig if self.priors_adj_missing_orig is not None else []) + [self.priors_adj_orig[i] for i in range(len(self.priors_adj_orig)) if not gene_mask[i]]
                self.priors_adj_orig = self.priors_adj_orig[gene_mask]

            if self.gene_pheno_combined_prior_Ys is not None:
                self.gene_pheno_combined_prior_Ys = self.gene_pheno_combined_prior_Ys[gene_mask,:]
            if self.gene_pheno_Y is not None:
                self.gene_pheno_Y = self.gene_pheno_Y[gene_mask,:]
            if self.gene_pheno_priors is not None:
                self.gene_pheno_priors = self.gene_pheno_priors[gene_mask,:]


        #for x in [self.priors, self.combined_prior_Ys, self.Y_orig, self.Y_w_orig, self.Y_fw_orig, self.priors_orig, self.combined_prior_Ys_orig]:
        #    if x is not None:
        #        x[:] = np.concatenate((x[gene_mask], x[~gene_mask]))

    #subset the current state of the class to a reduced set of gene sets
    def subset_gene_sets(self, subset_mask, keep_missing=True, ignore_missing=False, skip_V=False, skip_scale_factors=False):

        if subset_mask is None or np.sum(~subset_mask) == 0:
            return
        if self.gene_sets is None:
            return

        log("Subsetting gene sets", TRACE)

        remove_mask = np.logical_not(subset_mask)

        if ignore_missing:
            keep_missing = False

            if self.gene_sets is not None:
                if self.gene_sets_ignored is None:
                    self.gene_sets_ignored = []
                self.gene_sets_ignored = self.gene_sets_ignored + [self.gene_sets[i] for i in range(len(self.gene_sets)) if remove_mask[i]]

            if self.gene_set_labels is not None:
                if self.gene_set_labels_ignored is None:
                    self.gene_set_labels_ignored = []
                self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, self.gene_set_labels[remove_mask])

            if self.scale_factors is not None:
                if self.scale_factors_ignored is None:
                    self.scale_factors_ignored = np.array([])
                self.scale_factors_ignored = np.append(self.scale_factors_ignored, self.scale_factors[remove_mask])

            if self.mean_shifts is not None:
                if self.mean_shifts_ignored is None:
                    self.mean_shifts_ignored = np.array([])
                self.mean_shifts_ignored = np.append(self.mean_shifts_ignored, self.mean_shifts[remove_mask])

            if self.beta_tildes is not None:
                if self.beta_tildes_ignored is None:
                    self.beta_tildes_ignored = np.array([])
                self.beta_tildes_ignored = np.append(self.beta_tildes_ignored, self.beta_tildes[remove_mask])

            if self.p_values is not None:
                if self.p_values_ignored is None:
                    self.p_values_ignored = np.array([])
                self.p_values_ignored = np.append(self.p_values_ignored, self.p_values[remove_mask])

            if self.ses is not None:
                if self.ses_ignored is None:
                    self.ses_ignored = np.array([])
                self.ses_ignored = np.append(self.ses_ignored, self.ses[remove_mask])

            if self.z_scores is not None:
                if self.z_scores_ignored is None:
                    self.z_scores_ignored = np.array([])
                self.z_scores_ignored = np.append(self.z_scores_ignored, self.z_scores[remove_mask])

            if self.se_inflation_factors is not None:
                if self.se_inflation_factors_ignored is None:
                    self.se_inflation_factors_ignored = np.array([])
                self.se_inflation_factors_ignored = np.append(self.se_inflation_factors_ignored, self.se_inflation_factors[remove_mask])

            if self.gene_covariates is not None:
                if self.total_qc_metrics_ignored is None:
                    self.total_qc_metrics_ignored = self.total_qc_metrics[remove_mask,:]
                    self.mean_qc_metrics_ignored = self.mean_qc_metrics[remove_mask]
                else:
                    self.total_qc_metrics_ignored = np.vstack((self.total_qc_metrics_ignored, self.total_qc_metrics[remove_mask,:]))
                    self.mean_qc_metrics_ignored = np.append(self.mean_qc_metrics_ignored, self.mean_qc_metrics[remove_mask])

            #need to record how many ignored
            if self.X_orig is not None:
                if self.col_sums_ignored is None:
                    self.col_sums_ignored = np.array([])
                self.col_sums_ignored = np.append(self.col_sums_ignored, self.get_col_sums(self.X_orig[:,remove_mask]))

                gene_ignored_N = self.get_col_sums(self.X_orig[:,remove_mask], axis=1)
                if self.gene_ignored_N is None:
                    self.gene_ignored_N = gene_ignored_N
                else:
                    self.gene_ignored_N += gene_ignored_N
                if self.gene_N is not None:
                    self.gene_N -= gene_ignored_N

        elif keep_missing:
            self.gene_sets_missing = [self.gene_sets[i] for i in range(len(self.gene_sets)) if remove_mask[i]]

            if self.beta_tildes is not None:
                self.beta_tildes_missing = self.beta_tildes[remove_mask]
            if self.p_values is not None:
                self.p_values_missing = self.p_values[remove_mask]
            if self.z_scores is not None:
                self.z_scores_missing = self.z_scores[remove_mask]
            if self.ses is not None:
                self.ses_missing = self.ses[remove_mask]
            if self.se_inflation_factors is not None:
                self.se_inflation_factors_missing = self.se_inflation_factors[remove_mask]
            if self.beta_tildes_orig is not None:
                self.beta_tildes_missing_orig = self.beta_tildes_orig[remove_mask]
            if self.p_values_orig is not None:
                self.p_values_missing_orig = self.p_values_orig[remove_mask]
            if self.z_scores_orig is not None:
                self.z_scores_missing_orig = self.z_scores_orig[remove_mask]
            if self.ses_orig is not None:
                self.ses_missing_orig = self.ses_orig[remove_mask]

            if self.total_qc_metrics is not None:
                self.total_qc_metrics_missing = self.total_qc_metrics[remove_mask]

            if self.mean_qc_metrics is not None:
                self.mean_qc_metrics_missing = self.mean_qc_metrics[remove_mask]

            if self.inf_betas is not None:
                self.inf_betas_missing = self.inf_betas[remove_mask]

            if self.betas_uncorrected is not None:
                self.betas_uncorrected_missing = self.betas_uncorrected[remove_mask]

            if self.betas is not None:
                self.betas_missing = self.betas[remove_mask]
            if self.non_inf_avg_cond_betas is not None:
                self.non_inf_avg_cond_betas_missing = self.non_inf_avg_cond_betas[remove_mask]
            if self.non_inf_avg_postps is not None:
                self.non_inf_avg_postps_missing = self.non_inf_avg_postps[remove_mask]

            if self.inf_betas_orig is not None:
                self.inf_betas_missing_orig = self.inf_betas_orig[remove_mask]
            if self.betas_orig is not None:
                self.betas_missing_orig = self.betas_orig[remove_mask]
            if self.betas_uncorrected_orig is not None:
                self.betas_uncorrected_missing_orig = self.betas_uncorrected_orig[remove_mask]
            if self.non_inf_avg_cond_betas_orig is not None:
                self.non_inf_avg_cond_betas_missing_orig = self.non_inf_avg_cond_betas_orig[remove_mask]
            if self.non_inf_avg_postps_orig is not None:
                self.non_inf_avg_postps_missing_orig = self.non_inf_avg_postps_orig[remove_mask]

            if self.is_dense_gene_set is not None:
                self.is_dense_gene_set_missing = self.is_dense_gene_set[remove_mask]

            if self.gene_set_batches is not None:
                self.gene_set_batches_missing = self.gene_set_batches[remove_mask]

            if self.gene_set_labels is not None:
                self.gene_set_labels_missing = self.gene_set_labels[remove_mask]

            if self.ps is not None:
                self.ps_missing = self.ps[remove_mask]
            if self.sigma2s is not None:
                self.sigma2s_missing = self.sigma2s[remove_mask]


            if self.X_orig is not None:
                #store the removed gene sets for later
                if keep_missing:
                    self.X_orig_missing_gene_sets = self.X_orig[:,remove_mask]
                    self.mean_shifts_missing = self.mean_shifts[remove_mask]
                    self.scale_factors_missing = self.scale_factors[remove_mask]

        #now do the subsetting to keep

        if self.exp_gene_set_factors is not None:
            self.exp_gene_set_factors = self.exp_gene_set_factors[subset_mask,:]
        if self.gene_set_factor_gene_set_mask is not None:
            self.gene_set_factor_gene_set_mask = self.gene_set_factor_gene_set_mask[subset_mask]


        if self.beta_tildes is not None:
            self.beta_tildes = self.beta_tildes[subset_mask]
        if self.p_values is not None:
            self.p_values = self.p_values[subset_mask]
        if self.z_scores is not None:
            self.z_scores = self.z_scores[subset_mask]
        if self.ses is not None:
            self.ses = self.ses[subset_mask]
        if self.se_inflation_factors is not None:
            self.se_inflation_factors = self.se_inflation_factors[subset_mask]

        if self.beta_tildes_orig is not None:
            self.beta_tildes_orig = self.beta_tildes_orig[subset_mask]
        if self.p_values_orig is not None:
            self.p_values_orig = self.p_values_orig[subset_mask]
        if self.z_scores_orig is not None:
            self.z_scores_orig = self.z_scores_orig[subset_mask]
        if self.ses_orig is not None:
            self.ses_orig = self.ses_orig[subset_mask]


        if self.total_qc_metrics is not None:
            self.total_qc_metrics = self.total_qc_metrics[subset_mask]

        if self.mean_qc_metrics is not None:
            self.mean_qc_metrics = self.mean_qc_metrics[subset_mask]

        if self.inf_betas is not None:
            self.inf_betas = self.inf_betas[subset_mask]

        if self.betas_uncorrected is not None:
            self.betas_uncorrected = self.betas_uncorrected[subset_mask]

        if self.betas is not None:
            self.betas = self.betas[subset_mask]
        if self.non_inf_avg_cond_betas is not None:
            self.non_inf_avg_cond_betas = self.non_inf_avg_cond_betas[subset_mask]
        if self.non_inf_avg_postps is not None:
            self.non_inf_avg_postps = self.non_inf_avg_postps[subset_mask]

        if self.inf_betas_orig is not None:
            self.inf_betas_orig = self.inf_betas_orig[subset_mask]
        if self.betas_orig is not None:
            self.betas_orig = self.betas_orig[subset_mask]
        if self.betas_uncorrected_orig is not None:
            self.betas_uncorrected_orig = self.betas_uncorrected_orig[subset_mask]
        if self.non_inf_avg_cond_betas_orig is not None:
            self.non_inf_avg_cond_betas_orig = self.non_inf_avg_cond_betas_orig[subset_mask]
        if self.non_inf_avg_postps_orig is not None:
            self.non_inf_avg_postps_orig = self.non_inf_avg_postps_orig[subset_mask]

        if self.is_dense_gene_set is not None:
            self.is_dense_gene_set = self.is_dense_gene_set[subset_mask]

        if self.gene_set_batches is not None:
            self.gene_set_batches = self.gene_set_batches[subset_mask]

        if self.gene_set_labels is not None:
            self.gene_set_labels = self.gene_set_labels[subset_mask]

        if self.ps is not None:
            self.ps = self.ps[subset_mask]
        if self.sigma2s is not None:
            self.sigma2s = self.sigma2s[subset_mask]

        self.gene_sets = list(itertools.compress(self.gene_sets, subset_mask))
        self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)

        if self.X_phewas_beta is not None:
            self.X_phewas_beta = self.X_phewas_beta[:,subset_mask]                
        if self.X_phewas_beta_uncorrected is not None:
            self.X_phewas_beta_uncorrected = self.X_phewas_beta_uncorrected[:,subset_mask]                

        if self.X_orig is not None:
            #never update V; if it exists it will be updated below
            self._set_X(self.X_orig[:,subset_mask], self.genes, self.gene_sets, skip_V=True, skip_scale_factors=skip_scale_factors, skip_N=True)

        if self.X_orig_missing_genes is not None:
            #if we've already removed genes, then we need to remove the gene sets from them
            if keep_missing:
                self.X_orig_missing_genes_missing_gene_sets = self.X_orig_missing_genes[:,remove_mask]
            self.X_orig_missing_genes = self.X_orig_missing_genes[:,subset_mask]

        #need to update the scale factor for sigma2
        #sigma2 is always relative to just the non missing gene sets
        if self.sigma2 is not None:
            self.set_sigma(self.sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
        if self.p is not None:
            self.set_p(self.p)

    def _unsubset_gene_sets(self, skip_V=False, skip_scale_factors=False):
        if self.gene_sets_missing is None or self.X_orig_missing_gene_sets is None:
            return(np.array([True] * len(self.gene_sets)))

        log("Un-subsetting gene sets", TRACE)

        #need to update the scale factor for sigma2
        #sigma2 is always relative to just the non missing gene sets
        fraction_non_missing = self._get_fraction_non_missing()

        subset_mask = np.array([True] * len(self.gene_sets) + [False] * len(self.gene_sets_missing))

        self.gene_sets += self.gene_sets_missing
        self.gene_sets_missing = None
        self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)

        #if self.sigma2 is not None:
        #    old_sigma2 = self.sigma2
        #    self.set_sigma(self.sigma2 * fraction_non_missing, self.sigma_power, sigma2_osc=self.sigma2_osc)
        #    log("Changing sigma from %.4g to %.4g" % (old_sigma2, self.sigma2))
        #if self.p is not None:
        #    old_p = self.p
        #    self.set_p(self.p * fraction_non_missing)
        #    log("Changing p from %.4g to %.4g" % (old_p, self.p))

        if self.beta_tildes_missing is not None:
            self.beta_tildes = np.append(self.beta_tildes, self.beta_tildes_missing)
            self.beta_tildes_missing = None
        if self.p_values_missing is not None:
            self.p_values = np.append(self.p_values, self.p_values_missing)
            self.p_values_missing = None
        if self.z_scores_missing is not None:
            self.z_scores = np.append(self.z_scores, self.z_scores_missing)
            self.z_scores_missing = None
        if self.ses_missing is not None:
            self.ses = np.append(self.ses, self.ses_missing)
            self.ses_missing = None
        if self.se_inflation_factors_missing is not None:
            self.se_inflation_factors = np.append(self.se_inflation_factors, self.se_inflation_factors_missing)
            self.se_inflation_factors_missing = None

        if self.total_qc_metrics_missing is not None:
            self.total_qc_metrics = np.vstack((self.total_qc_metrics, self.total_qc_metrics_missing))
            self.total_qc_metrics_missing = None

        if self.mean_qc_metrics_missing is not None:
            self.mean_qc_metrics = np.append(self.mean_qc_metrics, self.mean_qc_metrics_missing)
            self.mean_qc_metrics_missing = None

        if self.beta_tildes_missing_orig is not None:
            self.beta_tildes_orig = np.append(self.beta_tildes_orig, self.beta_tildes_missing_orig)
            self.beta_tildes_missing_orig = None
        if self.p_values_missing_orig is not None:
            self.p_values_orig = np.append(self.p_values_orig, self.p_values_missing_orig)
            self.p_values_missing_orig = None
        if self.z_scores_missing_orig is not None:
            self.z_scores_orig = np.append(self.z_scores_orig, self.z_scores_missing_orig)
            self.z_scores_missing_orig = None
        if self.ses_missing_orig is not None:
            self.ses_orig = np.append(self.ses_orig, self.ses_missing_orig)
            self.ses_missing_orig = None

        if self.inf_betas_missing is not None:
            self.inf_betas = np.append(self.inf_betas, self.inf_betas_missing)
            self.inf_betas_missing = None

        if self.betas_uncorrected_missing is not None:
            self.betas_uncorrected = np.append(self.betas_uncorrected, self.betas_uncorrected_missing)
            self.betas_uncorrected_missing = None

        if self.betas_missing is not None:
            self.betas = np.append(self.betas, self.betas_missing)
            self.betas_missing = None
        if self.non_inf_avg_cond_betas_missing is not None:
            self.non_inf_avg_cond_betas = np.append(self.non_inf_avg_cond_betas, self.non_inf_avg_cond_betas_missing)
            self.non_inf_avg_cond_betas_missing = None
        if self.non_inf_avg_postps_missing is not None:
            self.non_inf_avg_postps = np.append(self.non_inf_avg_postps, self.non_inf_avg_postps_missing)
            self.non_inf_avg_postps_missing = None

        if self.inf_betas_missing_orig is not None:
            self.inf_betas_orig = np.append(self.inf_betas_orig, self.inf_betas_missing_orig)
            self.inf_betas_missing_orig = None
        if self.betas_missing_orig is not None:
            self.betas_orig = np.append(self.betas_orig, self.betas_missing_orig)
            self.betas_missing_orig = None
        if self.betas_uncorrected_missing_orig is not None:
            self.betas_uncorrected_orig = np.append(self.betas_uncorrected_orig, self.betas_uncorrected_missing_orig)
            self.betas_uncorrected_missing_orig = None
        if self.non_inf_avg_cond_betas_missing_orig is not None:
            self.non_inf_avg_cond_betas_orig = np.append(self.non_inf_avg_cond_betas_orig, self.non_inf_avg_cond_betas_missing_orig)
            self.non_inf_avg_cond_betas_missing_orig = None
        if self.non_inf_avg_postps_missing_orig is not None:
            self.non_inf_avg_postps_orig = np.append(self.non_inf_avg_postps_orig, self.non_inf_avg_postps_missing_orig)
            self.non_inf_avg_postps_missing_orig = None

        if self.X_orig_missing_gene_sets is not None:
            self.X_orig = sparse.hstack((self.X_orig, self.X_orig_missing_gene_sets), format="csc")
            self.X_orig_missing_gene_sets = None
            self.mean_shifts = np.append(self.mean_shifts, self.mean_shifts_missing)
            self.mean_shifts_missing = None
            self.scale_factors = np.append(self.scale_factors, self.scale_factors_missing)
            self.scale_factors_missing = None
            self.is_dense_gene_set = np.append(self.is_dense_gene_set, self.is_dense_gene_set_missing)
            self.is_dense_gene_set_missing = None
            self.gene_set_batches = np.append(self.gene_set_batches, self.gene_set_batches_missing)
            self.gene_set_batches_missing = None
            self.gene_set_labels = np.append(self.gene_set_labels, self.gene_set_labels_missing)
            self.gene_set_labels_missing = None


            if self.ps is not None:
                self.ps = np.append(self.ps, self.ps_missing)
                self.ps_missing = None
            if self.sigma2s is not None:
                self.sigma2s = np.append(self.sigma2s, self.sigma2s_missing)
                self.sigma2s_missing = None

        self._set_X(self.X_orig, self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=False)

        if self.X_orig_missing_genes_missing_gene_sets is not None:
            #if we've already removed genes, then we need to remove the gene sets from them
            self.X_orig_missing_genes = sparse.hstack((self.X_orig_missing_genes, self.X_orig_missing_genes_missing_gene_sets), format="csc")
            self.X_orig_missing_genes_missing_gene_sets = None

        return(subset_mask)


    #utility function to create a mapping from name to index in a list
    def _construct_map_to_ind(self, gene_sets):
        return dict([(gene_sets[i], i) for i in range(len(gene_sets))])

    #utility function to map names or indices to column indicies
    def _get_col(self, col_name_or_index, header_cols, require_match=True):
        try:
            if col_name_or_index is None:
                raise ValueError
            if int(col_name_or_index) <= 0:
                bail("All column ids specified as indices are 1-based")
            return(int(col_name_or_index) - 1)
        except ValueError:
            matching_cols = [i for i in range(0,len(header_cols)) if header_cols[i] == col_name_or_index]
            if len(matching_cols) == 0:
                if require_match:
                    bail("Could not find match for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
                else:
                    return None
            if len(matching_cols) > 1:
                bail("Found two matches for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
            return matching_cols[0]

    # inverse_matrix calculations
    def _invert_matrix(self,matrix_in):
        inv_matrix=np.linalg.inv(matrix_in) 
        return inv_matrix

    def _invert_sym_matrix(self,matrix_in):
        cho_factor = scipy.linalg.cho_factor(matrix_in)
        return scipy.linalg.cho_solve(cho_factor, np.eye(matrix_in.shape[0]))

    def _invert_matrix_old(self,matrix_in):
        sparsity=(matrix_in.shape[0]*matrix_in.shape[1]-matrix_in.count_nonzero())/(matrix_in.shape[0]*matrix_in.shape[1])
        log("Sparsity of matrix_in in invert_matrix %s" % sparsity, INFO)
        if sparsity>0.65:
            inv_matrix=np.linalg.inv(matrix_in.toarray()) # works efficiently for sparse matrix_in
        else:
            inv_matrix=sparse.linalg.inv(matrix_in) 
        return inv_matrix


##This function is for labelling clusters. Update it with your favorite LLM if desired
def query_lmm(query, auth_key=None):

    import requests

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer %s' % auth_key,
    }

    json_data = {
        #'model': 'gpt-3.5-turbo',
        'model': 'gpt-4o-mini',
        'messages': [
            {
                'role': 'user',
                'content': '%s' % query,
            },
        ],
    }
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data).json()
        if "choices" in response and len(response["choices"]) > 0 and "message" in response["choices"][0] and "content" in response["choices"][0]["message"]:
            return response["choices"][0]["message"]["content"]
        else:
            log("LMM response did not match the expected format; returning none. Response: %s" % response); 
            return None
    except Exception:
        log("LMM call failed; returning None"); 
        return None


def main():

    if not options.hide_opts:
        log("Python version: %s" % sys.version)
        log("Numpy version: %s" % np.__version__)
        log("Scipy version: %s" % scipy.__version__)
        log("Options: %s" % options)

    g = GeneSetData(background_prior=options.background_prior, batch_size=options.batch_size)

    #g.read_X(options.X_in)
    #y = []
    #for line in open("c"):
    #    a = line.strip().split()
    #    y.append(a)
    #y = np.array(y)
    #bail("")

    sigma2_cond = options.sigma2_cond


    if sigma2_cond is not None:
        #map it with the scale factor
        g.set_sigma(options.sigma2_ext, options.sigma_power, convert_sigma_to_internal_units=False)
        sigma2_cond = g.get_sigma2()
        g.set_sigma(None, g.sigma_power)
    elif options.sigma2_ext is not None:
        g.set_sigma(options.sigma2_ext, options.sigma_power, convert_sigma_to_internal_units=True)
        log("Setting sigma=%.4g (given external=%.4g) " % (g.get_sigma2(), g.get_sigma2(convert_sigma_to_external_units=True)))
    elif options.sigma2 is not None:
        g.set_sigma(options.sigma2, options.sigma_power, convert_sigma_to_internal_units=False)
    elif options.top_gene_set_prior:
        g.set_sigma(g.convert_prior_to_var(options.top_gene_set_prior, options.num_gene_sets_for_prior if options.num_gene_sets_for_prior is not None else len(g.gene_sets), options.frac_gene_sets_for_prior), options.sigma_power, convert_sigma_to_internal_units=True)
        if options.frac_gene_sets_for_prior == 1:
            #in this case sigma2_cond was specified, not sigma2
            sigma2_cond = g.get_sigma2()
            log("Setting sigma_cond=%.4g (external=%.4g) given top of %d gene sets prior of %.4g" % (g.get_sigma2(), g.get_sigma2(convert_sigma_to_external_units=True), options.num_gene_sets_for_prior, options.top_gene_set_prior))
            g.set_sigma(None, g.sigma_power)
        else:
            log("Setting sigma=%.4g (external=%.4g) given top of %d gene sets prior of %.4g" % (g.get_sigma2(), g.get_sigma2(convert_sigma_to_external_units=True), options.num_gene_sets_for_prior, options.top_gene_set_prior))
                        
    #sigma calculations
    if options.const_sigma:
        options.sigma_power = 2

    if options.update_hyper.lower() == "both":
        options.update_hyper_p = True
        options.update_hyper_sigma = True
    elif options.update_hyper.lower() == "p":
        options.update_hyper_p = True
        options.update_hyper_sigma = False
    elif options.update_hyper.lower() == "sigma2" or options.update_hyper.lower() == "sigma":
        options.update_hyper_p = False
        options.update_hyper_sigma = True
    elif options.update_hyper.lower() == "none":
        options.update_hyper_p = False
        options.update_hyper_sigma = False
    else:
        bail("Invalid value for --update-hyper (both, p, sigma2, or none)")

    if options.gene_map_in:
        g.read_gene_map(options.gene_map_in, options.gene_map_orig_gene_col, options.gene_map_orig_gene_col)
    if options.gene_loc_file:
        g.init_gene_locs(options.gene_loc_file)

    def _default_for_gene_list():
        options.ols = True
        if options.positive_controls_all_in is None:
            warn("Specified positive controls without --positive-controls-all-in; therefore using all genes in gene sets as negatives. This may result in inflated enrichments")
            options.add_all_genes = True

    if run_factor and expand_gene_sets and (options.add_gene_sets_by_enrichment_p is not None or options.add_gene_sets_by_naive is not None or options.add_gene_sets_by_gibbs is not None):
        #we are going to use the machinery of betas/gibbs to expand the gene list
        #even though internally this will be stored as betas/priors/etc, we will not be factoring this
        #these will be overwritten during the factoring
        options.positive_controls_list = options.anchor_genes
        _default_for_gene_list()

    #we don't need to read in any matrices if we are anchoring to phenotypes, because those will use the (later) phewas files
    #we need to use it if we are anchoring to a gene only if we are going to use the phewas results to factor
    extend_for_gene = run_factor and use_phewas_for_factoring and options.anchor_genes is not None and (options.add_gene_sets_by_enrichment_p is not None or options.add_gene_sets_by_naive is not None or options.add_gene_sets_by_gibbs is not None)

    if (not run_factor or not use_phewas_for_factoring or extend_for_gene) and ((run_factor and expand_gene_sets and extend_for_gene) or run_huge or run_beta_tilde or run_sigma or run_beta or run_priors or run_naive_priors or run_gibbs or run_factor):

        if not extend_for_gene and options.gene_bfs_in:
            g.read_Y(gene_bfs_in=options.gene_bfs_in,show_progress=not options.hide_progress, gene_bfs_id_col=options.gene_bfs_id_col, gene_bfs_log_bf_col=options.gene_bfs_log_bf_col, gene_bfs_combined_col=options.gene_bfs_combined_col, gene_bfs_prob_col=options.gene_bfs_prob_col, gene_bfs_prior_col=options.gene_bfs_prior_col, gene_covs_in=options.gene_covs_in, hold_out_chrom=options.hold_out_chrom)
        elif extend_for_gene or options.gwas_in or options.exomes_in or options.positive_controls_in or options.positive_controls_list is not None:
            if not use_phewas_for_factoring and options.gwas_in is None and options.exomes_in is None:
                _default_for_gene_list()

            g.read_Y(gwas_in=options.gwas_in,show_progress=not options.hide_progress, gwas_chrom_col=options.gwas_chrom_col, gwas_pos_col=options.gwas_pos_col, gwas_p_col=options.gwas_p_col, gwas_beta_col=options.gwas_beta_col, gwas_se_col=options.gwas_se_col, gwas_n_col=options.gwas_n_col, gwas_n=options.gwas_n, gwas_units=options.gwas_units, gwas_freq_col=options.gwas_freq_col, gwas_filter_col=options.gwas_filter_col, gwas_filter_value=options.gwas_filter_value, gwas_locus_col=options.gwas_locus_col, gwas_ignore_p_threshold=options.gwas_ignore_p_threshold, gwas_low_p=options.gwas_low_p, gwas_high_p=options.gwas_high_p, gwas_low_p_posterior=options.gwas_low_p_posterior, gwas_high_p_posterior=options.gwas_high_p_posterior, detect_low_power=options.gwas_detect_low_power, detect_high_power=options.gwas_detect_high_power, detect_adjust_huge=options.gwas_detect_adjust_huge, learn_window=options.learn_window, closest_gene_prob=options.closest_gene_prob, max_closest_gene_prob=options.max_closest_gene_prob, scale_raw_closest_gene=options.scale_raw_closest_gene, cap_raw_closest_gene=options.cap_raw_closest_gene, cap_region_posterior=options.cap_region_posterior, scale_region_posterior=options.scale_region_posterior, phantom_region_posterior=options.phantom_region_posterior, allow_evidence_of_absence=options.allow_evidence_of_absence, correct_huge=options.correct_huge, gws_prob_true=options.gene_zs_gws_prob_true, max_closest_gene_dist=options.max_closest_gene_dist, signal_window_size=options.signal_window_size, signal_min_sep=options.signal_min_sep, signal_max_logp_ratio=options.signal_max_logp_ratio, credible_set_span=options.credible_set_span, min_n_ratio=options.min_n_ratio, max_clump_ld=options.max_clump_ld, exomes_in=options.exomes_in, exomes_gene_col=options.exomes_gene_col, exomes_p_col=options.exomes_p_col, exomes_beta_col=options.exomes_beta_col, exomes_se_col=options.exomes_se_col, exomes_n_col=options.exomes_n_col, exomes_n=options.exomes_n, exomes_units=options.exomes_units, exomes_low_p=options.exomes_low_p, exomes_high_p=options.exomes_high_p, exomes_low_p_posterior=options.exomes_low_p_posterior, exomes_high_p_posterior=options.exomes_high_p_posterior, positive_controls_in=options.positive_controls_in, positive_controls_id_col=options.positive_controls_id_col, positive_controls_prob_col=options.positive_controls_prob_col, positive_controls_default_prob=options.positive_controls_default_prob, positive_controls_has_header=options.positive_controls_has_header, positive_controls_list=options.positive_controls_list, positive_controls_all_in=options.positive_controls_all_in, positive_controls_all_id_col=options.positive_controls_all_id_col, positive_controls_all_has_header=options.positive_controls_all_has_header, gene_loc_file=options.gene_loc_file_huge if options.gene_loc_file_huge is not None else options.gene_loc_file, gene_covs_in=options.gene_covs_in, hold_out_chrom=options.hold_out_chrom, exons_loc_file=options.exons_loc_file_huge, min_var_posterior=options.min_var_posterior, s2g_in=options.s2g_in, s2g_chrom_col=options.s2g_chrom_col, s2g_pos_col=options.s2g_pos_col, s2g_gene_col=options.s2g_gene_col, s2g_prob_col=options.s2g_prob_col, s2g_normalize_values=options.s2g_normalize_values, credible_sets_in=options.credible_sets_in, credible_sets_id_col=options.credible_sets_id_col, credible_sets_chrom_col=options.credible_sets_chrom_col, credible_sets_pos_col=options.credible_sets_pos_col, credible_sets_ppa_col=options.credible_sets_ppa_col)
        elif options.gene_percentiles_in:
            g.read_Y(gene_percentiles_in=options.gene_percentiles_in,show_progress=not options.hide_progress, gene_percentiles_id_col=options.gene_percentiles_id_col, gene_percentiles_value_col=options.gene_percentiles_value_col, gene_percentiles_higher_is_better=options.gene_percentiles_higher_is_better, gene_percentiles_top_posterior=options.top_posterior, gene_covs_in=options.gene_covs_in, hold_out_chrom=options.hold_out_chrom)
        elif options.gene_zs_in:
            g.read_Y(gene_zs_in=options.gene_zs_in,show_progress=not options.hide_progress, gene_zs_id_col=options.gene_zs_id_col, gene_zs_value_col=options.gene_zs_value_col, gws_threshold=options.gene_zs_gws_threshold, gws_prob_true=options.gene_zs_gws_prob_true, max_mean_posterior=options.gene_zs_max_mean_posterior, gene_covs_in=options.gene_covs_in, hold_out_chrom=options.hold_out_chrom)
        #else:
        #    bail("Need --gwas-in or --exomes-in or --gene-bfs-in or --gene-percentiles-in or --gene-zs-in")

    if not run_huge:

        gene_set_ids = None
        if run_factor:
            
            #here we are only getting the IDs we'll keep
            #it will save us time in reading in gene sets below in read_X since we can skip gene sets not in these files
            if options.gene_set_stats_in is not None and not use_phewas_for_factoring:
                gene_set_ids = g.read_gene_set_statistics(options.gene_set_stats_in, stats_id_col=options.gene_set_stats_id_col, stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col, stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col, stats_p_col=options.gene_set_stats_p_col, stats_se_col=options.gene_set_stats_se_col, stats_beta_col=options.gene_set_stats_beta_col, stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col, ignore_negative_exp_beta=options.ignore_negative_exp_beta, max_gene_set_p=options.max_gene_set_read_p, min_gene_set_beta=options.min_gene_set_read_beta, min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected, return_only_ids=True)
            elif use_phewas_for_factoring:
                if options.gene_set_phewas_stats_in is None:
                    bail("Need --gene-set-phewas-stats-in")
                gene_set_ids = g.read_gene_set_phewas_statistics(options.gene_set_phewas_stats_in, stats_id_col=options.gene_set_phewas_stats_id_col, stats_pheno_col=options.gene_set_phewas_stats_pheno_col, stats_beta_col=options.gene_set_phewas_stats_beta_col, stats_beta_uncorrected_col=options.gene_set_phewas_stats_beta_uncorrected_col, min_gene_set_beta=options.min_gene_set_read_beta, min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected, return_only_ids=True, phenos_to_match=options.anchor_phenos)

            if gene_set_ids is not None:
                log("Will read %d gene sets" % (len(gene_set_ids)), DEBUG)
                

        #read in the matrices
        if options.X_in is not None or options.X_list is not None or options.Xd_in is not None or options.Xd_list is not None:
            filter_gene_set_p = options.filter_gene_set_p
            force_reread = False
            while True:
                orig_sigma2 = g.sigma2

                skip_betas = (run_huge or run_beta_tilde or run_sigma or run_factor) and (not run_beta and not run_priors and not run_naive_priors and not run_gibbs and (not run_factor or options.gene_set_stats_in is not None or use_phewas_for_factoring))

                genes_to_inc = None
                if run_factor and use_phewas_for_factoring:
                    genes_to_inc = options.anchor_genes
                    options.max_num_gene_sets = None
                    
                g.read_X(options.X_in, Xd_in=options.Xd_in, X_list=options.X_list, Xd_list=options.Xd_list, V_in=options.V_in, min_gene_set_size=options.min_gene_set_size, max_gene_set_size=options.max_gene_set_size, only_ids=gene_set_ids, only_inc_genes=genes_to_inc, fraction_inc_genes=options.add_gene_sets_by_fraction, add_all_genes=options.add_all_genes, prune_gene_sets=options.prune_gene_sets, prune_deterministically=options.prune_deterministically, x_sparsify=options.x_sparsify, add_ext=options.add_ext, add_top=options.add_top, add_bottom=options.add_bottom, filter_negative=options.filter_negative, threshold_weights=options.threshold_weights, cap_weights=options.cap_weights, permute_gene_sets=options.permute_gene_sets, max_gene_set_p=options.max_gene_set_read_p, filter_gene_set_p=filter_gene_set_p, increase_filter_gene_set_p=options.increase_filter_gene_set_p, max_num_gene_sets_initial=options.max_num_gene_sets_initial, max_num_gene_sets=options.max_num_gene_sets, skip_betas=skip_betas, run_logistic=not options.linear, max_for_linear=options.max_for_linear, filter_gene_set_metric_z=options.filter_gene_set_metric_z, initial_p=options.p_noninf, initial_sigma2=g.sigma2, initial_sigma2_cond=sigma2_cond, sigma_power=options.sigma_power, sigma_soft_threshold_95=options.sigma_soft_threshold_95, sigma_soft_threshold_5=options.sigma_soft_threshold_5, run_gls=False, run_corrected_ols=not options.ols, correct_betas_mean=options.correct_betas_mean, correct_betas_var=options.correct_betas_var, gene_loc_file=options.gene_loc_file, gene_cor_file=options.gene_cor_file, gene_cor_file_gene_col=options.gene_cor_file_gene_col, gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col, update_hyper_p=options.update_hyper_p, update_hyper_sigma=options.update_hyper_sigma, batch_all_for_hyper=options.batch_all_for_hyper, first_for_hyper=options.first_for_hyper, first_max_p_for_hyper=options.first_max_p_for_hyper, first_for_sigma_cond=options.first_for_sigma_cond, sigma_num_devs_to_top=options.sigma_num_devs_to_top, p_noninf_inflate=options.p_noninf_inflate, batch_separator=options.batch_separator, ignore_genes=set(options.ignore_genes), file_separator=options.file_separator, max_num_burn_in=options.max_num_burn_in, max_num_iter_betas=options.max_num_iter_betas, min_num_iter_betas=options.min_num_iter_betas, num_chains_betas=options.num_chains_betas, r_threshold_burn_in_betas=options.r_threshold_burn_in_betas, use_max_r_for_convergence_betas=options.use_max_r_for_convergence_betas, max_frac_sem_betas=options.max_frac_sem_betas, max_allowed_batch_correlation=options.max_allowed_batch_correlation, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas, betas_trace_out=options.betas_trace_out, show_progress=not options.hide_progress, skip_V=(options.max_gene_set_read_p is not None), force_reread=force_reread)

                if gene_set_ids is not None:
                    break
                if options.min_num_gene_sets is None or filter_gene_set_p is None or filter_gene_set_p >= 1 or g.gene_sets is None or len(g.gene_sets) >= options.min_num_gene_sets:
                    break
                if filter_gene_set_p < 1:
                    fraction_to_increase = float(options.min_num_gene_sets) / len(g.gene_sets)
                    assert(fraction_to_increase > 1)
                    #add in a fudge factor
                    filter_gene_set_p *= fraction_to_increase * 1.2
                    if filter_gene_set_p > 1:
                        filter_gene_set_p = 1
                    log("Only read in %d gene sets; scaled --filter-gene-set-p to %.3g and re-reading gene sets" % (len(g.gene_sets), filter_gene_set_p))
                    force_reread = True
                    #reset sigma
                    g.set_sigma(orig_sigma2, g.sigma_power)
                else:
                    break
                    
        if options.X_out:
            g.write_X(options.X_out)
        if options.Xd_out:
            g.write_Xd(options.Xd_out)
        if options.V_out:
            g.write_V(options.V_out)

        if run_sim:
            g.run_sim(sigma2=g.sigma2, p=g.p, sigma_power=g.sigma_power, log_bf_noise_sigma_mult=options.sim_log_bf_noise_sigma_mult, treat_sigma2_as_sigma2_cond=False)

        run_gibbs_for_factor = False
        run_beta_for_factor = False

        if run_factor is not None and options.const_gene_set_beta is not None:
            g.beta_tildes = np.full(len(g.gene_sets), options.const_gene_set_beta)
        elif options.gene_set_stats_in is not None and not use_phewas_for_factoring:
            g.read_gene_set_statistics(options.gene_set_stats_in, stats_id_col=options.gene_set_stats_id_col, stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col, stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col, stats_p_col=options.gene_set_stats_p_col, stats_se_col=options.gene_set_stats_se_col, stats_beta_col=options.gene_set_stats_beta_col, stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col, ignore_negative_exp_beta=options.ignore_negative_exp_beta, max_gene_set_p=options.max_gene_set_read_p, min_gene_set_beta=options.min_gene_set_read_beta, min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected)
        elif run_beta_tilde or run_sigma or run_beta or run_priors or run_naive_priors or run_gibbs or (run_factor and not use_phewas_for_factoring and (options.anchor_gene_set is not None or not factor_gene_set_x_pheno)) or (run_factor and (options.add_gene_sets_by_naive or options.add_gene_sets_by_gibbs)) or run_sim:
            if run_factor:
                run_beta_for_factor = True
                if not run_naive_factor:
                    run_gibbs_for_factor = True

            g.calculate_gene_set_statistics(max_gene_set_p=options.filter_gene_set_p, run_gls=False, run_logistic=not options.linear, max_for_linear=options.max_for_linear, run_corrected_ols=not options.ols, use_sampling_for_betas=options.use_sampling_for_betas, correct_betas_mean=options.correct_betas_mean, correct_betas_var=options.correct_betas_var, gene_loc_file=options.gene_loc_file, gene_cor_file=options.gene_cor_file, gene_cor_file_gene_col=options.gene_cor_file_gene_col, gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col)

        if run_factor:
            if options.gene_set_phewas_stats_in is not None:
                g.read_gene_set_phewas_statistics(options.gene_set_phewas_stats_in, stats_id_col=options.gene_set_phewas_stats_id_col, stats_pheno_col=options.gene_set_phewas_stats_pheno_col, stats_beta_col=options.gene_set_phewas_stats_beta_col, stats_beta_uncorrected_col=options.gene_set_phewas_stats_beta_uncorrected_col, min_gene_set_beta=options.min_gene_set_read_beta, min_gene_set_beta_uncorrected=options.min_gene_set_read_beta_uncorrected)

            if options.gene_phewas_bfs_in:
                g.read_gene_phewas_bfs(gene_phewas_bfs_in=options.gene_phewas_bfs_in,gene_phewas_bfs_id_col=options.gene_phewas_bfs_id_col, gene_phewas_bfs_pheno_col=options.gene_phewas_bfs_pheno_col, anchor_genes=options.anchor_genes, anchor_phenos=options.anchor_phenos, gene_phewas_bfs_log_bf_col=options.gene_phewas_bfs_log_bf_col, gene_phewas_bfs_combined_col=options.gene_phewas_bfs_combined_col, gene_phewas_bfs_prior_col=options.gene_phewas_bfs_prior_col, min_value=options.min_gene_phewas_read_value)

        if run_sigma or ((run_beta or run_priors or run_naive_priors or run_gibbs or run_beta_for_factor) and g.sigma2 is None):
            g.calculate_sigma(options.sigma_power, options.chisq_threshold, options.chisq_dynamic, options.desired_intercept_difference)

        if options.cross_val:
            g.run_cross_val(options.cross_val_num_explore_each_direction, folds=options.cross_val_folds, cross_val_max_num_tries=options.cross_val_max_num_tries, p=options.p_noninf if g.p is None else g.p, max_num_burn_in=options.max_num_burn_in, max_num_iter=options.max_num_iter_betas, min_num_iter=options.min_num_iter_betas, num_chains=options.num_chains_betas, run_logistic=not options.linear, max_for_linear=options.max_for_linear, run_corrected_ols=not options.ols, r_threshold_burn_in=options.r_threshold_burn_in_betas, use_max_r_for_convergence=options.use_max_r_for_convergence_betas, max_frac_sem=options.max_frac_sem_betas, gauss_seidel=options.gauss_seidel_betas, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas)

        #gene set betas
        if run_factor is not None and options.const_gene_set_beta is not None:
            g.betas = np.full(len(g.gene_sets), options.const_gene_set_beta)
            g.betas_uncorrected = np.full(len(g.gene_sets), options.const_gene_set_beta)
        elif (not run_factor or not use_phewas_for_factoring) and options.gene_set_betas_in:
            g.read_betas(options.gene_set_betas_in)
        elif run_beta or run_priors or run_naive_priors or run_gibbs or run_beta_for_factor:
            #if False:
            #    g.calculate_inf_betas(update_hyper_sigma=options.update_hyper_sigma)
            #update hyper was done above while while reading x
            g.calculate_non_inf_betas(options.p_noninf if g.p is None else g.p, max_num_burn_in=options.max_num_burn_in, max_num_iter=options.max_num_iter_betas, min_num_iter=options.min_num_iter_betas, num_chains=options.num_chains_betas, r_threshold_burn_in=options.r_threshold_burn_in_betas, use_max_r_for_convergence=options.use_max_r_for_convergence_betas, max_frac_sem=options.max_frac_sem_betas, max_allowed_batch_correlation=options.max_allowed_batch_correlation, gauss_seidel=options.gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas, pre_filter_batch_size=options.pre_filter_batch_size, pre_filter_small_batch_size=options.pre_filter_small_batch_size, betas_trace_out=options.betas_trace_out)

        #priors
        if run_priors:
            g.calculate_priors(max_gene_set_p=options.filter_gene_set_p, num_gene_batches=options.priors_num_gene_batches, correct_betas_mean=options.correct_betas_mean, correct_betas_var=options.correct_betas_var, gene_loc_file=options.gene_loc_file, gene_cor_file=options.gene_cor_file, gene_cor_file_gene_col=options.gene_cor_file_gene_col, gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col, p_noninf=options.p_noninf if g.p is None else g.p, run_logistic=not options.linear, max_for_linear=options.max_for_linear, adjust_priors=options.adjust_priors, max_num_burn_in=options.max_num_burn_in, max_num_iter=options.max_num_iter_betas, min_num_iter=options.min_num_iter_betas, num_chains=options.num_chains_betas, r_threshold_burn_in=options.r_threshold_burn_in_betas, use_max_r_for_convergence=options.use_max_r_for_convergence_betas, max_frac_sem=options.max_frac_sem_betas, max_allowed_batch_correlation=options.max_allowed_batch_correlation, gauss_seidel=options.gauss_seidel_betas, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas)
        elif run_naive_priors or (run_naive_factor and not use_phewas_for_factoring):
            g.calculate_naive_priors(adjust_priors=options.adjust_priors)

        if run_factor is not None and options.const_gene_log_bf is not None:
            g.Y = np.full(len(g.genes), options.const_gene_log_bf)
            g.combined_prior_Ys = np.full(len(g.genes), options.const_gene_log_bf)
        elif run_gibbs or run_gibbs_for_factor:
            g.run_gibbs(min_num_iter=options.min_num_iter, max_num_iter=options.max_num_iter, num_chains=options.num_chains, num_mad=options.num_mad, r_threshold_burn_in=options.r_threshold_burn_in, max_frac_sem=options.max_frac_sem, use_max_r_for_convergence=options.use_max_r_for_convergence, p_noninf=options.p_noninf if g.p is None else g.p, increase_hyper_if_betas_below=options.increase_hyper_if_betas_below, update_huge_scores=options.update_huge_scores, top_gene_prior=options.top_gene_prior, max_num_burn_in=options.max_num_burn_in, max_num_iter_betas=options.max_num_iter_betas, min_num_iter_betas=options.min_num_iter_betas, num_chains_betas=options.num_chains_betas, r_threshold_burn_in_betas=options.r_threshold_burn_in_betas, use_max_r_for_convergence_betas=options.use_max_r_for_convergence_betas, max_frac_sem_betas=options.max_frac_sem_betas, use_mean_betas=not options.use_sampled_betas_in_gibbs, sparse_frac_gibbs=options.sparse_frac_gibbs, sparse_max_gibbs=options.sparse_max_gibbs, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas, pre_filter_batch_size=options.pre_filter_batch_size, pre_filter_small_batch_size=options.pre_filter_small_batch_size, max_allowed_batch_correlation=options.max_allowed_batch_correlation, gauss_seidel=options.gauss_seidel, gauss_seidel_betas=options.gauss_seidel_betas, num_gene_batches=options.priors_num_gene_batches, num_batches_parallel=options.gibbs_num_batches_parallel, max_mb_X_h=options.gibbs_max_mb_X_h, initial_linear_filter=options.initial_linear_filter, adjust_priors=options.adjust_priors, correct_betas_mean=options.correct_betas_mean, correct_betas_var=options.correct_betas_var, gene_set_stats_trace_out=options.gene_set_stats_trace_out, gene_stats_trace_out=options.gene_stats_trace_out, betas_trace_out=options.betas_trace_out)

    if options.gene_set_stats_out:
        g.write_gene_set_statistics(options.gene_set_stats_out, max_no_write_gene_set_beta=options.max_no_write_gene_set_beta, max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected)

    if options.gene_stats_out:
        g.write_gene_statistics(options.gene_stats_out)

    if options.gene_gene_set_stats_out:
        g.write_gene_gene_set_statistics(options.gene_gene_set_stats_out, max_no_write_gene_gene_set_beta=options.max_no_write_gene_gene_set_beta, write_filter_beta_uncorrected=options.use_beta_uncorrected_for_gene_gene_set_write_filter)

    if options.gene_set_overlap_stats_out:
        g.write_gene_set_overlap_statistics(options.gene_set_overlap_stats_out)

    if options.gene_covs_out:
        g.write_gene_covariates(options.gene_covs_out)

    if options.gene_effectors_out:
        g.write_gene_effectors(options.gene_effectors_out)

    if run_phewas:
        #run the phewas

        bfs_to_use = options.run_phewas_from_gene_phewas_stats_in

        if options.gene_phewas_bfs_in is not None and bfs_to_use == options.gene_phewas_bfs_in and g.num_gene_phewas_filtered == 0:
            #we can skip reading if we are using the same file as previously read and we didn't threshold that file
            bfs_to_use = None

        g.run_phewas(gene_phewas_bfs_in=bfs_to_use,gene_phewas_bfs_id_col=options.gene_phewas_bfs_id_col, gene_phewas_bfs_pheno_col=options.gene_phewas_bfs_pheno_col, gene_phewas_bfs_log_bf_col=options.gene_phewas_bfs_log_bf_col, gene_phewas_bfs_combined_col=options.gene_phewas_bfs_combined_col, gene_phewas_bfs_prior_col=options.gene_phewas_bfs_prior_col, max_num_burn_in=options.max_num_burn_in, max_num_iter=options.max_num_iter_betas, min_num_iter=options.min_num_iter_betas, num_chains=options.num_chains_betas, r_threshold_burn_in=options.r_threshold_burn_in_betas, use_max_r_for_convergence=options.use_max_r_for_convergence_betas, max_frac_sem=options.max_frac_sem_betas, gauss_seidel=options.gauss_seidel_betas, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas)

        if options.phewas_stats_out:
            g.write_phewas_statistics(options.phewas_stats_out)

    if run_factor:

        if expand_gene_sets:

            if options.add_gene_sets_by_naive is not None or options.add_gene_sets_by_gibbs is not None:
                assert(g.betas_uncorrected is not None)
                #need to use external ones here
                g.subset_gene_sets(g.betas_uncorrected / g.scale_factors > (options.add_gene_sets_by_gibbs if options.add_gene_sets_by_gibbs is not None else options.add_gene_sets_by_naive))
                if len(g.gene_sets) == 0:
                    bail("Subsetting gene sets by %s removed all gene sets; try reducing threshold" % ("gibbs" if options.add_gene_sets_by_gibbs is not None else "naive"))
                else:
                    log("Pruning by %s resulted in %d gene sets; try reducing threshold" % ("gibbs" if options.add_gene_sets_by_gibbs is not None else "naive", len(g.gene_sets)), DEBUG)

        #need to update signature here and replace

        if options.anchor_gene_set:
            gene_or_pheno_filter_value = options.gene_set_pheno_filter_value
        elif factor_gene_set_x_pheno:
            gene_or_pheno_filter_value = options.pheno_filter_value
        else:
            gene_or_pheno_filter_value = options.gene_filter_value

        g.run_factor(max_num_factors=options.max_num_factors, phi=options.phi, alpha0=options.alpha0, beta0=options.beta0, gene_set_filter_value=options.gene_set_filter_value, gene_or_pheno_filter_value=gene_or_pheno_filter_value, pheno_prune_value=options.factor_prune_phenos_val, pheno_prune_number=options.factor_prune_phenos_num, gene_prune_value=options.factor_prune_genes_val, gene_prune_number=options.factor_prune_genes_num, gene_set_prune_value=options.factor_prune_gene_sets_val, gene_set_prune_number=options.factor_prune_gene_sets_num, anchor_pheno_mask=g.anchor_pheno_mask, anchor_gene_mask=g.anchor_gene_mask, anchor_any_pheno=options.anchor_any_pheno, anchor_any_gene=options.anchor_any_gene, anchor_gene_set=options.anchor_gene_set, run_transpose=not options.no_transpose, min_lambda_threshold=options.min_lambda_threshold, lmm_auth_key=options.lmm_auth_key)


    if options.factors_out is not None:
        g.write_matrix_factors(options.factors_out)

    if options.factors_anchor_out is not None:
        g.write_matrix_factors(options.factors_anchor_out, write_anchor_specific=True)

    if options.gene_set_clusters_out is not None or options.gene_clusters_out is not None or options.pheno_clusters_out is not None:
        g.write_clusters(options.gene_set_clusters_out, options.gene_clusters_out, options.pheno_clusters_out)

    if options.gene_set_anchor_clusters_out is not None or options.gene_anchor_clusters_out is not None or options.pheno_anchor_clusters_out is not None:
        g.write_clusters(options.gene_set_anchor_clusters_out, options.gene_anchor_clusters_out, options.pheno_anchor_clusters_out, write_anchor_specific=True)


    if options.gene_pheno_stats_out is not None:
        g.write_gene_pheno_statistics(options.gene_pheno_stats_out, min_value_to_print=options.max_no_write_gene_pheno)

    if options.factor_phewas_from_gene_phewas_stats_in is not None:

        if g.num_factors() > 0:

            bfs_to_use = options.factor_phewas_from_gene_phewas_stats_in

            if (options.gene_phewas_bfs_in is not None and bfs_to_use == options.gene_phewas_bfs_in) or (options.run_phewas_from_gene_phewas_stats_in is not None and bfs_to_use == options.run_phewas_from_gene_phewas_stats_in) and g.num_gene_phewas_filtered == 0:
                #we can skip reading if we are using the same file as previously read and we didn't threshold that file
                bfs_to_use = None

            g.run_phewas(gene_phewas_bfs_in=bfs_to_use,gene_phewas_bfs_id_col=options.gene_phewas_bfs_id_col, gene_phewas_bfs_pheno_col=options.gene_phewas_bfs_pheno_col, gene_phewas_bfs_log_bf_col=options.gene_phewas_bfs_log_bf_col, gene_phewas_bfs_combined_col=options.gene_phewas_bfs_combined_col, gene_phewas_bfs_prior_col=options.gene_phewas_bfs_prior_col, max_num_burn_in=options.max_num_burn_in, max_num_iter=options.max_num_iter_betas, min_num_iter=options.min_num_iter_betas, num_chains=options.num_chains_betas, r_threshold_burn_in=options.r_threshold_burn_in_betas, use_max_r_for_convergence=options.use_max_r_for_convergence_betas, max_frac_sem=options.max_frac_sem_betas, gauss_seidel=options.gauss_seidel_betas, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas, run_for_factors=True, batch_size=500, min_gene_factor_weight=options.factor_phewas_min_gene_factor_weight)
            if options.factor_phewas_stats_out:
                g.write_factor_phewas_statistics(options.factor_phewas_stats_out)
        else:
            log("No factors; not performing factor phewas")


    if options.params_out:
        g.write_params(options.params_out)

if __name__ == '__main__':

    #profiler = cProfile.Profile()
    #profiler.enable()

    #cProfile.run('main()')
    main()

    #profiler.disable()
    #profiler.dump_stats('output.prof')


