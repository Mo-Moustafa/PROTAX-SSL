#!/bin/bash

#SBATCH --account="aip-gwtaylor"
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=z-runs_outputs/PBert_cls%j.out
#SBATCH --error=z-runs_outputs/PBert_cls%j.err

module load cuda/12.9 cudnn/9.13.1.26
source pssl.venv/bin/activate


id="3444465"
MODEL="bert"
DIST_SCALING="log"

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

echo "Traininig Configuration: $TC_JSON"
echo "Experiment Details: $exp_details"
echo "----------------------------------------------"
echo "Taxonomy Used: $data_dir"
echo "----------------------------------------------"
echo "Running on model_${id}"

#---------------------------


test_mode="test"
python -m scripts.process_seqs "$data_dir" "$test_mode" models/model_${id}.npz "$TC_JSON" "$exp_details" "${id}"



# /usr/bin/time -v python -m scripts.process_seqs_bert "$data_dir" "$test_mode" models/model_${id}.npz "$TC_JSON" "$exp_details" "${id}"
#   sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,AllocTRES%40
