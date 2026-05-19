#!/bin/bash

#SBATCH --account="aip-gwtaylor"
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=z-runs_outputs/train%j.out
#SBATCH --error=z-runs_outputs/train%j.err

module load cuda/12.9 cudnn/9.13.1.26
source pssl.venv/bin/activate


# id="3444465"
MODEL="bert"
DIST_SCALING="log"
scalings_dir="models/scalings/plain.npz"

LR=0.05
BATCH_SIZE=512
NUM_EPOCHS=10
L2=0.0

DATA="mycoai"

#---------------------------

TC_JSON=$(printf '{"model":"%s","learning_rate":%s,"batch_size":%s,"num_epochs":%s,"dist_scaling":"%s","l2":%s}' \
  "$MODEL" "$LR" "$BATCH_SIZE" "$NUM_EPOCHS" "$DIST_SCALING" "$L2")

exp_details="${MODEL} ($DATA)"
if [ "$MODEL" = "og" ]; then
  data_dir="datasets/$DATA/fasta"
else
  data_dir="datasets/$DATA/embeddings"
fi

echo "Run Number $SLURM_JOB_ID"
echo "Traininig Configuration: $TC_JSON"
echo "Experiment Details: $exp_details"
echo "----------------------------------------------"
echo "Taxonomy Used: $data_dir"
echo "Scalings Used: $scalings_dir"
echo "----------------------------------------------"

#---------------------------

python -m scripts.train --data_dir "$data_dir" --scalings_dir "$scalings_dir" --tc "$TC_JSON" --exp "$exp_details" --id "$SLURM_JOB_ID" --continue_training False
# test_mode="test"
# python -m scripts.process_seqs_bert "$data_dir" "$test_mode" models/model_${SLURM_JOB_ID}.npz "$TC_JSON" "$exp_details" "${SLURM_JOB_ID}"



# echo "Running on model_${id}"
# python -m scripts.train --data_dir "$data_dir" --scalings_dir "$scalings_dir" --tc "$TC_JSON" --exp "$exp_details" --id "${id}" --continue_training True
# test_mode="test"
# python -m scripts.process_seqs_bert "$data_dir" "$test_mode" models/model_${id}.npz "$TC_JSON" "$exp_details" "${id}"

# /usr/bin/time -v python -m scripts.train --data_dir "$data_dir" --scalings_dir "$scalings_dir" --tc "$TC_JSON" --exp "$exp_details" --id "$SLURM_JOB_ID" --continue_training False
#   sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,AllocTRES%40
