#!/bin/bash

declare -a algorithms=( hc_cont tabu_cont )

for alg in "${algorithms[@]}"
do
  echo "$alg"
  Rscript bnlearn/"$alg".R
done
