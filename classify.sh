
source your_virtual_env.venv/bin/activate
export HWLOC_HIDE_ERRORS=1
export PYTHONUNBUFFERED=1

id=""
MODEL="og"
DIST_SCALING="log"

LR=5e-0
BATCH_SIZE=2048
NUM_EPOCHS=20
L2=0.0

DATA="your_data_name"

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

#--------------------------- Script to run classification on a test set

# test_mode="test"
# python -m scripts.process_seqs "$data_dir" "$test_mode" models/model_${id}.npz "$TC_JSON" "$exp_details" "${id}"