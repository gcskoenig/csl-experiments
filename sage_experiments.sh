#!/bin/bash

Rscript datagen/datagen_env.R 1000000

Rscript datagen/datagen_pcalg.R 1000000

python sample_targets.py

declare -a datasets=( dag_s_0.2 dag_s_0.3 dag_s_0.4 dag_sm_0.1 dag_sm_0.15 dag_sm_0.2 dag_m_0.04 dag_m_0.06 dag_m_0.08
                    dag_l_0.02 dag_l_0.03 dag_l_0.04)

for data in "${datasets[@]}"
do
  echo "$data"
  python sage_cont.py --data "$data"
done
