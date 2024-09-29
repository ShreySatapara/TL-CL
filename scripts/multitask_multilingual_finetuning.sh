gpu_ids=$1
seq_no=seq1
port_no=$2
lr_scheduler=constant
mkdir -p ../outputs/multitask_multilingual_finetuning_10_epochs_${lr_scheduler}_mt5_base/training_logs

CUDA_VISIBLE_DEVICES=$gpu_ids  accelerate launch --main_process_port $port_no ../multitask_training.py \
    --root_data_dir ../data \
    --model_name_or_path google/mt5-base \
    --root_output_dir ../outputs/multitask_multilingual_finetuning_10_epochs_${lr_scheduler}_mt5_base \
    --num_train_epochs 10 \
    --per_device_batch_size 8 \
    --learning_rate 1e-4 \
    --results_dict ../results/multitask_multilingual_finetuning_10_epochs_${lr_scheduler}_mt5_base \
    --logging_steps 100 \
    --do_train \
    --do_eval \
    --task_sequence $seq_no \
    --gradient_accumulation_steps 2 \
    --lr_scheduler $lr_scheduler \
    --task_sequence_file ../task_sequences.json 2>&1 | tee ../outputs/multitask_multilingual_finetuning_10_epochs_${lr_scheduler}_mt5_base/training_logs/$seq_no.log