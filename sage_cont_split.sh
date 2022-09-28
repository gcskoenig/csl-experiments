#!/bin/bash

declare -a runs=( 1 2 3 4 5 )

for run in "${runs[@]}"
do
  echo "$run"
  python sage_cont.py --data "dag_l_0.03" --model "lm" --runs 1 --sage_seed 42 --test True --run "$run"
done
