#!/bin/bash

declare -a datasets=( dag_s_0.2 dag_s_0.3 dag_s_0.4 dag_sm_0.1 dag_sm_0.15 dag_sm_0.2 dag_m_0.04 dag_m_0.06 dag_m_0.08 )

for data in "${datasets[@]}"
do
  echo "$data"
  python ai_via_timing.py --data "$data"
done
