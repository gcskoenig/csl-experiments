#!/bin/bash

declare -a algorithms=( hc_cont tabu_cont )

for alg in "${algorithms[@]}"
do
  echo "$data"
  Rscript bnlearn/"$alg".R
done
