#!/bin/bash
iter_num=3
for i in $(seq 1 $iter_num); do
    username="HF_USERNAME"
    name="SELM-Llama-3-8B-Instruct"
    fraction=$((61135/(iter_num)))
    training_dataset="HuggingFaceH4/ultrafeedback_binarized"
    model_name_or_path="$username/${name}-iter-$((i-1))"
    dataset_mixer="{'updated':'$username/${name}-dataset_iter_$i','original':'$training_dataset'}"
    dataset_splits=("train_prefs[$((fraction*(i-1))):$((fraction*i))]","test_prefs")
    hub_model_id="${name}-iter-$i"
    output_dir="data/$hub_model_id"
    if [ "$i" -eq 1 ]; then
        learning_rate=5e-7
        model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
    elif [ "$i" -eq 2 ]; then
        learning_rate=3e-7
    else
        learning_rate=1e-7
    fi
    python scripts/online_feedback.py recipes/llama3-selm/selm_config_full.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer dataset_splits=$dataset_splits || exit 1
    ACCELERATE_LOG_LEVEL=info /home/aiscuser/.local/bin/accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_selm.py recipes/llama3-selm/selm_config_full.yaml learning_rate=$learning_rate alpha=$alpha model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir || exit 1
done