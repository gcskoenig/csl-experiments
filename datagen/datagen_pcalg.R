#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# Generating conditionally linear Gaussian graphs and sampling synthetic data in accordance to them for an average degree of args[2]

# project folder, this file is supposed to be in first level of project folder
# setwd("~/Desktop/csl-experiments/")

# create folders to store results about DGPs and adjacency matrices
dir.create("data")
dir.create("data/dgps")
dir.create("data/true_amat")

# install packages if necessary 
#if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
#BiocManager::install("graph")
#install.packages('pcalg')

# load package 'pcalg'
library(pcalg)

# set seed
set.seed(44)

# number of nodes for small to large graph
size <- c(10, 20, 50, 100)

# names of graphs (add prob to graph name)
size_tokens <- c("s", "sm", "m", "l")

# sample size of sampled data sets
n <- strtoi(args[1])
degree <- strtoi(args[2])

for (i in c(1:4)){
    d <- size[i]
    token <- size_tokens[i]
    
    # randomly generate DAG
    graph <- r.gauss.pardag(d, prob=degree/d, top.sort = FALSE, normalize = TRUE,
                            lbe = 0.1, ube = 1, neg.coef = TRUE, labels = as.character(1:d),
                            lbv = 0.5, ubv = 1)
    
    # retrieve and store info about DGP (weight matrix and error variance)
    proba <- round(degree/d,digits=5) 
    weights <- graph$weight.mat()
    error_var <- graph$err.var()
    weights_filename <- paste("data/dgps/dag_", token,"_", proba ,"_weights.csv", sep="")
    write.csv(weights, weights_filename, row.names = FALSE)
    variance_filename <- paste("data/dgps/dag_", token, "_", proba, "_error_var.csv", sep="")
    write.csv(error_var, variance_filename, row.names = FALSE)
    
    # retrieve Boolean adjacency matrix and store it
    amat <- as(graph, "matrix")
    amat_name <- paste("data/true_amat/dag_", token, "_", proba, ".csv", sep="")
    write.csv(amat, amat_name, row.names = FALSE)
    
    # create and store data
    data <- graph$simulate(n)
    filename <- paste("data/dag_", token, "_", proba, ".csv", sep="")
    write.csv(data, filename, row.names = FALSE)
    
}
