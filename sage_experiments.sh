#!/bin/bash

Rscript datagen/datagen_pcalg.R 1000000 2

python sample_targets.py

declare -a datasets=( dag_s_0.2 dag_sm_0.1 dag_m_0.04 dag_l_0.02 )

for data in "${datasets[@]}"
do
  echo "$data"
  python sage_cont.py --data "$data"
done
