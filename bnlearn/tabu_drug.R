# Bayesian network structure learning using tabu algorithm (as implemented in bnlearn)
# setwd("~/Desktop/csl-experiments")
# directory bnlearn supposed to exist
# dir.create("bnlearn")
dir.create("real-world-experiments/results")
dir.create("real-world-experiments/results/tabu")

#install.packages("bnlearn")
library("bnlearn")

# set seed
set.seed(1902)


# initiate data frame to store metadata like runtime
table <- data.frame(matrix(ncol = 4, nrow = 0))
col_names <- c("Graph", "Sample Size", "Algorithm", "Runtime in s")
colnames(table) <- col_names


filename <- paste("real-world-experiments/drug_consumption_preprocessed.csv", sep="")
df <- read.csv(filename)
  
# as.factor() required for bnlearn.tabu()
cat_vars = c('Gender', 'Education', 'Country', 'Ethnicity', 'Nicotine')
for (j in cat_vars){
  df[,j] <- as.factor(df[,j]) 
}
  
# structure learning and wall time
runtime <- system.time({ bn <- tabu(df) })
runtime <- runtime["elapsed"]
table[nrow(table) + 1,] = c('Drug', 'full', "tabu", runtime)

# adjacency matrix
adj_mat <- amat(bn)
amat_file <- paste("real-world-experiments/results/tabu/drug_consumption.csv", sep="")
write.csv(adj_mat, file=amat_file, row.names = FALSE)

# save table
write.csv(table,"real-world-experiments/results/tabu/runtime_drug_consumption.csv", row.names = FALSE)

