# When working with pods, one has to send command to all tpus workers
TPU_NAME=$1
COMMAND=$2
ZONE=europe-west4-a

echo $COMMAND

# TODO: wrap this in tmux in order for command not to be killed upon lost of ssh connection.
gcloud alpha compute tpus tpu-vm ssh --strict-host-key-checking=no ${TPU_NAME} --zone ${ZONE} --worker=all --command="$COMMAND" -- -t

# Example to run t5_c4_span_corruption
#  - run setup vms: sh eleutherai/scripts/run_on_all_vms.sh multitask "$(cat eleutherai/scripts/setup_vm.sh)"
#  - run t5_c4_span_corruption: sh eleutherai/scripts/run_on_all_vms.sh multitask "cd code/architecture-objective; git pull; sh bigscience/scripts/launch_command_in_tmux.sh \"bash eleutherai/scripts/pretrain.sh enc_dec_c4_span_corruption\""
#  - kill zombie process: sh eleutherai/scripts/run_on_all_vms.sh multitask "killall -u {USER}"
