gpu_ids=$1
port_no=$2
seq_no=$3
training_method=sequential_finetuning
cl_method=sequential_finetuning_25_epochs_mt5_base


mkdir -p ../outputs/$cl_method/training_logs

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch --main_process_port $port_no ../main.py \
    --model_name_or_checkpoint_path google/mt5-small \
    --training_method sequential_finetuning \
    --task_sequence $seq_no \
    --num_train_epochs_per_task 25 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --root_output_dir ../outputs/$cl_method \
    --root_data_dir ../data \
    --logging_steps 100 \
    --task_sequence_file ../task_sequences.json \
    --do_eval \
    --do_train \
    --root_results_dir ../results/$cl_method > ../outputs/$cl_method/training_logs/$seq_no.log

