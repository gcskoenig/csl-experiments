# Bayesian network structure learning using tabu algorithm (as implemented in bnlearn)
setwd("~/Desktop/sage_cg_local")
dir.create("bnlearn/results")
dir.create("bnlearn/results/tabu")

#install.packages("bnlearn")
library("bnlearn")

# set seed
set.seed(1902)

# load data and adapt
bike <- paste("data/bike.csv", sep="")
# only for bike dataset
bike <- bike[2:12]
bike[1:11] <- lapply(bike[1:11], as.numeric)
bike <- as.data.frame(bike)

bank <- paste("data/bank", sep="")
credit <- paste("data/credit", sep="")



# to loop through different data sets
graphs_cont <- bike

# initiate data frame to store metadata like runtime
table <- data.frame(matrix(ncol = 4, nrow = 0))
col_names <- c("Graph", "Sample Size", "Algorithm", "Runtime in s")
colnames(table) <- col_names

    
# structure learning and wall time
runtime <- system.time({ bn <- tabu(bike) })
runtime <- runtime["elapsed"]
# table[nrow(table) + 1,] = c(i, sample_size, "tabu", runtime)
    
# adjacency matrix
adj_mat <- amat(bn)
amat_file <- paste("bnlearn/results/tabu/bike_amat.csv", sep="")
write.csv(adj_mat, file=amat_file, row.names = FALSE)
  


# save table
#write.csv(table,"bnlearn/results/tabu/runtime_data_cont.csv", row.names = FALSE)

