#!/bin/bash

JOB1=$(sbatch --parsable epoch1.sh)
echo "Submitted epoch 1: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 epoch2.sh)
echo "Submitted epoch 2: $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 epoch3.sh)
echo "Submitted epoch 3: $JOB3 (depends on $JOB2)"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 epoch4.sh)
echo "Submitted epoch 4: $JOB4 (depends on $JOB3)"

JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 epoch5.sh)
echo "Submitted epoch 5: $JOB5 (depends on $JOB4)"
