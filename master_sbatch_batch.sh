#!/bin/bash

for i in $(seq 1 "$1")
do
	sbatch sbatch_submit.sh "$2"
done
