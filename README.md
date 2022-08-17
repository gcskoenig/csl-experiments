# Causal Structure Learning for SAGE Estimation

Code accompanying the paper 'Causal Structure Learning for Efficient SAGE Estimation'

## Summary

R code to generate data from R environments (.RDA files) downloaded from the
[bnlearn repository](https://www.bnlearn.com/bnrepository/) as well as completely synthetic data. R files to infer
graphs from (semi-)synthetic as well as real-world data. Benchmark of graph learning with respect to learned 
d-separations. Experiment files to fit models and infer SAGE values for the features of the respective models.
Experiment file to exploit d-separations/independeces in SAGE inference. File to test the spared time of latter approach
as opposed to standard SAGE estimation.

## Replicate

To replicate the study, clone this repository, so that all directories adhere the requirements of the different R and
Python scripts.

```
git clone [link]
```

To make use of the RFI package, clone it into this the directory of this (cloned) repository and install.

```
git clone [link]
```

and 

```
pip install -e rfi
```

where rfi is the path to the directory of the cloned RFI repository. Now proceed as explained below:

## Data

We used the software R (version 4.12) to generate the data (if necessary) and conduct the causal structure learning.

To generate semi-synthetic categorical data from .RDA files (located in ~/datagen/envs/) execute:

```
RScript datagen/datagen_env.R
```

To generate synthetic linear Gaussian data using the R package 'pcalg' (link) execute: 

```
RScript datagen/datagen_pcalg.R
```

The real-world datasets are available from: Link


## Causal Structure Learning

To conduct causal structure learning, we again relied on the software R (version 4.10). To execute the respective 
learning algorithm, choose the file accordingly. For semi-synthetic categorical data, the files are named by the
respective algorithm only (e.g. tabu.R for a greedy search algorithm using TABU search heuristic). For the synthetic
continuous data, the R files have the addendum '_cont' after the algorithm name (e.g. tabu_cont.R). The application of 
the algorithms to the employed real-world data is done using the files with the addendum '_real' (e.g. tabu_real.R). 
Each file covers all data files that pertain to the respective descriptions and each .csv file has to exist and be in 
the corresponding directory according to the structure in this repository.

Application of tabu search algorithm to every file containing synthetic linear Gaussian data:

```
RScript bnlearn/tabu_cont.R
```

## Evaluation of Causal Structure Learning

Evaluation of causal structure learning is done with respect to d-separations between a set of potential explanatory 
variables and a dedicated target. Since targets are (mostly) sampled at random for the (semi-)synthetic data, you need
to execute the sampling of targets first:

```
python sample_targets.py
```

Then you can execute:

```
python graph_evaluation.py
```

## SAGE Inference

For all experiments conducted using Python we used version 3.8.13 (check!) (PUT IN BEGINNING)

Use respective experiment files, sage_cont.py for synthetic continuous data, sage_discrete.py for semi-synthetic 
discrete data and sage_real.py for real-world data. We use the SAGE implementation from (link). E.g.

```
python sage_cont.py
```

Now convert the results (based on intermediate results from the SAGE inference) to csl-sage/experimental results. This
is done by setting inferred summands of the SAGE representation:

(equation)

to zero given X_j dsep Y given coalition. This is done post-hoc to the SAGE results in order to save runtime during the
experiments but also implemented in the original rfi package to be used in practice. To this end, execute:

```
python sage_to_csl_sage.py
```  

The file fct_eval.py serves the purpose to evaluate which part of the SAGE approximation that is potentially skipped 
when a d-separation can be found. To replicate the results, execute

```
python fct_eval.py
```


## Visualization

Successively execute files from visualization folder.