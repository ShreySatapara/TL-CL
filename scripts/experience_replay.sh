conda activate new_transformers_torch
gpu_ids=$1
port_no=$2
per_task_examples=$3
seq_no_1=$4
training_method=sequential_finetuning
cl_method=experience_replay_25_epochs_${per_task_examples}_examples_per_task

mkdir -p ../outputs/$cl_method/training_logs

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch --main_process_port $port_no ../main.py \
    --model_name_or_checkpoint_path google/mt5-small \
    --training_method sequential_finetuning \
    --task_sequence $seq_no_1 \
    --num_train_epochs_per_task 25 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --root_output_dir ../outputs/$cl_method \
    --root_data_dir ../data \
    --logging_steps 100 \
    --task_sequence_file ../task_sequences.json \
    --do_train \
    --do_eval \
    --root_results_dir ../results/$cl_method \
    --do_experience_reply \
    --task_wise_examples_for_ER $per_task_examples > ../outputs/$cl_method/training_logs/$seq_no_1.log

# sleep 20
sleep 20
# remove unnecessary intermidiate checkpoints
rm -rf ../outputs/$cl_method/*/*/checkpoints*