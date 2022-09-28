# Bayesian network structure learning using tabu algorithm (as implemented in bnlearn)
# setwd("~/Desktop/csl-experiments")
# directory bnlearn supposed to exist
# dir.create("bnlearn")
dir.create("bnlearn/results")
dir.create("bnlearn/results/tabu")

#install.packages("bnlearn")
library("bnlearn")

# set seed
set.seed(1902)

# to loop through different data sets
graphs_discrete <- c("alarm", "asia", "hepar", "sachs")
sample_sizes <- c(1000, 10000, 100000, 1000000)

# initiate data frame to store metadata like runtime
table <- data.frame(matrix(ncol = 4, nrow = 0))
col_names <- c("Graph", "Sample Size", "Algorithm", "Runtime in s")
colnames(table) <- col_names

for (i in graphs_discrete){
  # load data
  filename <- paste("data/", i, ".csv", sep="")
  df <- read.csv(filename)
  
  # as.factor() required for bnlearn.tabu()
  for (j in colnames(df)){
    df[,j] <- as.factor(df[,j]) 
  }
  
  for (sample_size in sample_sizes){
    
    # sample sample_size data from the dataset
    data_fit <- df[1:sample_size,]
    
    # structure learning and wall time
    runtime <- system.time({ bn <- tabu(data_fit) })
    runtime <- runtime["elapsed"]
    table[nrow(table) + 1,] = c(i, sample_size, "tabu", runtime)
    
    # adjacency matrix
    adj_mat <- amat(bn)
    amat_file <- paste("bnlearn/results/tabu/", i, "_", sample_size, "_obs.csv", sep="")
    write.csv(adj_mat, file=amat_file, row.names = FALSE)
  }
}

# save table
write.csv(table,"bnlearn/results/tabu/runtime_data_discrete.csv", row.names = FALSE)

