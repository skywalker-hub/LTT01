CONFIG_FILE="configs/example_config.yaml"
NUM_GPUS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    eval/eval_LT_Tuning.py \
    $CONFIG_FILE
