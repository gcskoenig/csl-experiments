#!/bin/bash

declare -a degrees=( 2 3 4 )

for deg in "${degrees[@]}"
do
  echo "$deg"
  Rscript datagen/datagen_pcalg.R 1000 "$deg"
done
