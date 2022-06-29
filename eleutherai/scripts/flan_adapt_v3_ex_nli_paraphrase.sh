python3 -c "import jax; print(jax.host_count(), jax.device_count(), jax.local_device_count())"

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
#CHECKPOINT_STEP=$2
# EXPERIMENT_NAME=$ORIGINAL_EXPERIMENT_NAME"_t0_adapt_"$CHECKPOINT_STEP
# CHECKPOINT_DIR="gs://bigscience-t5x/arch_objective_exps_v2/$ORIGINAL_EXPERIMENT_NAME/checkpoint_$CHECKPOINT_STEP"
CHECKPOINT_DIR="gs://neo-datasets/zphang/multitask/imported_checkpoints/c_dec_c4_full_lm_bs2048/checkpoint_131072/checkpoint_131072/"
# MODEL_DIR="gs://bigscience-t5x/arch_objective_exps_v2/$EXPERIMENT_NAME"
MODEL_DIR="gs://neo-datasets/zphang/multitask/runs/testing/v3_ex_nli_paraphrase"

# directory where the T5X repo is cloned.
T5X_DIR="/home/connor/code/architecture-objective"
export PYTHONPATH=${T5X_DIR}/bigscience/gins:${PYTHONPATH}
export PYTHONPATH=/home/connor/code/FLAN/:${PYTHONPATH}
export TFDS_DATA_DIR=gs://neo-datasets/zphang/multitask/tfds/v1

# Logs
LOGS_PATH="/home/connor/logs"
mkdir -p $LOGS_PATH

#if [[ $ORIGINAL_EXPERIMENT_NAME == c_dec* ]]
#then
#  GIN_FILE=c_dec_t0_adapt.gin
#fi
#if [[ $ORIGINAL_EXPERIMENT_NAME == nc_dec* ]]
#then
#  GIN_FILE=nc_dec_t0_adapt.gin
#fi
#if [[ $ORIGINAL_EXPERIMENT_NAME == enc_dec* ]]
#then
#  GIN_FILE=enc_dec_t0_adapt.gin
#fi
#if [[ $GIN_FILE == "" ]]
#then
#  echo "Incorrect experiment name $ORIGINAL_EXPERIMENT_NAME, does not start with c_dec/nc_dec/enc_dec"
#  exit
#fi

GIN_FILE=c_dec_flan_adapt.gin
GIN_FILE=eleutherai/gins/$GIN_FILE
echo "Running the following config: $GIN_FILE" 2>&1 | tee $LOGS_PATH/pretrain.txt

TFDS_DATA_DIR=gs://neo-datasets/zphang/multitask/tfds/v1 python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="$GIN_FILE" \
  --gin.INITIAL_CHECKPOINT_PATH="'${CHECKPOINT_DIR}'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  2>&1 | tee -a $LOGS_PATH/pretrain.txt

# bash bigscience/scripts/t0_adapt.sh c_dec_c4_full_lm_bs_128 420000
