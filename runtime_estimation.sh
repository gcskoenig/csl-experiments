#!/bin/bash

declare -a datasets=( dag_s_0.2 dag_sm_0.1 dag_m_0.04 dag_l_0.02 
			dag_s_0.3 dag_sm_0.15 dag_m_0.06 dag_l_0.03
			dag_s_0.4 dag_sm_0.2 dag_m_0.08 dag_l_0.04
			)

for data in "${datasets[@]}"
do
  echo "$data"
  ai_via_timing.py --data "$data"--model lm --size 10000 --runs 5 --no_order 100
  ai_via_timing.py --data "$data"--model rf --size 10000 --runs 5 --no_order 100
done
