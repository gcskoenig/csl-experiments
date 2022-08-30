#!/bin/bash

declare -a datasets=( dag_s_0.2 dag_sm_0.1 dag_m_0.04 )

for data in "${datasets[@]}"
do
  echo "$data"
  python sage_cont.py --data "$data" --model "rf"
done
