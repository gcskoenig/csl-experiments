#!/bin/bash



declare -a runs=( 1 2 3 4 5 )

for run in "${runs[@]}"
do
  echo "$run"
  python sage_cont.py --data "dag_s_0.3" --model "rf" --runs 1 --sage_seed 42 --test True --run "$run" --size 100 --orderings 10
done
