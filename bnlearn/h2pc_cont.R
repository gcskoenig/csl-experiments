# Bayesian network structure learning using h2pc algorithm (as implemented in bnlearn)
#setwd("~/Desktop/thesis_code")
dir.create("bnlearn/results")
dir.create("bnlearn/results/h2pc")

#install.packages("bnlearn")
library("bnlearn")

# set seed
set.seed(1902)

# to loop through different data sets
graphs_cont <- c("dag_s", "dag_sm", "dag_m", "dag_l")
sample_sizes <- c("1000", "10000", "100000", "1000000", "2000000")

# initiate data frame to store metadata like runtime
table <- data.frame(matrix(ncol = 4, nrow = 0))
col_names <- c("Graph", "Sample Size", "Algorithm", "Runtime in s")
colnames(table) <- col_names

for (i in graphs_cont){
  # load data
  filename <- paste("data/", i, ".csv", sep="")
  df <- read.csv(filename)
  
  for (sample_size in sample_sizes){
    
    data_fit <- df[sample(nrow(df), sample_size), ]
    
    # if conditions only necessary for the respective graphs (unused)
    if (i == "healthcare"){
      # as.factor() required for bnlearn.h2pc()
      for (j in c("A", "C", "H")){
        df[,j] <- as.factor(df[,j]) 
      }
    }
    
    if (i == "mehra"){
      # as.factor() required for bnlearn.h2pc()
      for (j in c("Region", "Zone", "Type", "Season", "Year", "Month", "Day", "Hour")){
        df[,j] <- as.factor(df[,j]) 
      }
    }
    
    if (i == "sangiovese"){
      # as.factor() required for bnlearn.h2pc()
      for (j in c("Treatment")){
        df[,j] <- as.factor(df[,j]) 
      }
    }
    
    # structure learning and wall time
    runtime <- system.time({ bn <- h2pc(data_fit) })
    runtime <- runtime["elapsed"]
    table[nrow(table) + 1,] = c(i, sample_size, "h2pc", runtime)
    
    # adjacency matrix
    adj_mat <- amat(bn)
    amat_file <- paste("bnlearn/results/h2pc/", i, "_", sample_size, "_obs.csv", sep="")
    write.csv(adj_mat, file=amat_file, row.names = FALSE)
  }
}

# save table
write.csv(table,"bnlearn/results/h2pc/runtime_data_cont.csv", row.names = FALSE)

