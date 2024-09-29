gpu_ids=$1
port_no=$2
seq_no_1=$3
training_method=mad_x
cl_method=mad_x_cl

mkdir -p ../outputs/mad_x_individual

declare -a task_seq=("cls_en" "cls_es" "cls_ar" "cls_hi" "nli_en" "nli_es" "nli_ar" "nli_hi" "qa_en" "qa_es" "qa_ar" "qa_hi" "summ_en" "summ_es" "summ_ar" "summ_hi")


for seq_no in "${task_seq[@]}"
do
    CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch --main_process_port $port_no ../main.py \
        --model_name_or_checkpoint_path google/mt5-base \
        --training_method sequential_finetuning \
        --task_sequence $seq_no \
        --num_train_epochs_per_task 10 \
        --per_device_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 \
        --root_output_dir ../outputs/individual_finetuning_mt5_base_10_epochs \
        --root_data_dir ../data \
        --logging_steps 100 \
        --task_sequence_file ../task_sequences.json \
        --do_eval \
        --do_train \
        --root_results_dir ../results/individual_finetuning_mt5_base_10_epochs > ../outputs/individual_finetuning_mt5_base_10_epochs/training_logs/$seq_no.log 
done