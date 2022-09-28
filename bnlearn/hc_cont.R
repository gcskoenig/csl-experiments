# Bayesian network structure learning using hc algorithm (as implemented in bnlearn)
# setwd("~/Desktop/csl-experiments/")
dir.create("bnlearn/results")
dir.create("bnlearn/results/hc")

#install.packages("bnlearn")
library("bnlearn")

# set seed
set.seed(1902)

# to loop through different data sets
graphs_cont <- c("dag_s_0.2", "dag_s_0.3", "dag_s_0.4", "dag_sm_0.1", "dag_sm_0.15", "dag_sm_0.2",
                 "dag_m_0.04", "dag_m_0.06", "dag_m_0.08", "dag_l_0.02", "dag_l_0.03", "dag_l_0.04")
sample_sizes <- c(1000, 10000, 100000, 1000000)

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
      # as.factor() required for bnlearn.hc()
      for (j in c("A", "C", "H")){
        df[,j] <- as.factor(df[,j]) 
      }
    }
    
    if (i == "mehra"){
      # as.factor() required for bnlearn.hc()
      for (j in c("Region", "Zone", "Type", "Season", "Year", "Month", "Day", "Hour")){
        df[,j] <- as.factor(df[,j]) 
      }
    }
    
    if (i == "sangiovese"){
      # as.factor() required for bnlearn.hc()
      for (j in c("Treatment")){
        df[,j] <- as.factor(df[,j]) 
      }
    }
    
    # structure learning and wall time
    runtime <- system.time({ bn <- hc(data_fit) })
    runtime <- runtime["elapsed"]
    table[nrow(table) + 1,] = c(i, sample_size, "hc", runtime)
    
    # adjacency matrix
    adj_mat <- amat(bn)
    amat_file <- paste("bnlearn/results/hc/", i, "_", sample_size, "_obs.csv", sep="")
    write.csv(adj_mat, file=amat_file, row.names = FALSE)
  }
}

# save table
write.csv(table,"bnlearn/results/hc/runtime_data_cont.csv", row.names = FALSE)

