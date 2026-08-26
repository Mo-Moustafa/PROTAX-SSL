
source your_virtual_env.venv/bin/activate
export HWLOC_HIDE_ERRORS=1
export PYTHONUNBUFFERED=1

id="0"
continue_training="false"

MODEL="og"    # Options are: og, bert, mamba, hybrid_lin, mlp
DIST_SCALING="log"  # Options are: z-score, log, power, hybrid, log-z, none
scalings_dir="models/scalings/plain.npz"

LR=5e-0
DECAY_LR="true"
BATCH_SIZE=2048
NUM_EPOCHS=20
L2=0.0

DATA="your_data_name"

#---------------------------

TC_JSON=$(printf '{"model":"%s","learning_rate":%s,"decay_lr":%s,"batch_size":%s,"num_epochs":%s,"dist_scaling":"%s","l2":%s,"continue_training":%s}' \
  "$MODEL" "$LR" "$DECAY_LR" "$BATCH_SIZE" "$NUM_EPOCHS" "$DIST_SCALING" "$L2" "$continue_training")

exp_details="${MODEL} ($DATA)"
if [ "$MODEL" = "og" ]; then
  data_dir="datasets/$DATA/fasta"
else
  data_dir="datasets/$DATA/embeddings"
fi

echo "Run Model Number $id"
echo "Traininig Configuration: $TC_JSON"
echo "Experiment Details: $exp_details"
echo "----------------------------------------------"
echo "Taxonomy Used: $data_dir"
echo "Scalings Used: $scalings_dir"
echo "----------------------------------------------"

#--------------------------- Script to run training

# python -m scripts.train --data_dir "$data_dir" --scalings_dir "$scalings_dir" --tc "$TC_JSON" --exp "$exp_details" --id "${id}"