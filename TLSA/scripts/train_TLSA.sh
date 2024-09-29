conda activate new_transformers_torch

gpu_ids=$1
port_no=$2
seq_1=$3
importance=$4
encoder_ids_for_lang=0,2
encoder_ids_for_task=1,3,4,5,6,7
decoder_ids_for_lang=5,7
decoder_ids_for_task=0,1,2,3,4,6



epochs=25

output_dir=../../outputs_online_ewc/peft_with_ewc_${epochs}_${importance}_enl_${encoder_ids_for_lang}_ent_${encoder_ids_for_task}_del_${decoder_ids_for_lang}_det_${decoder_ids_for_task}
mkdir -p $output_dir/training_logs

seq=$seq_1

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch --main_process_port $port_no main.py \
    --model_name_or_checkpoint_path google/mt5-small \
    --task_sequence $seq \
    --num_train_epochs_per_task $epochs \
    --per_device_batch_size 8 \
    --learning_rate 1e-4 \
    --root_output_dir $output_dir \
    --root_results_dir ../../results_online_ewc/peft_with_ewc_${epochs}_${importance}_enl_${encoder_ids_for_lang}_ent_${encoder_ids_for_task}_del_${decoder_ids_for_lang}_det_${decoder_ids_for_task} \
    --root_data_dir ../../data \
    --logging_steps 100 \
    --do_eval \
    --do_train \
    --task_sequence_file ../../task_sequences.json \
    --adapter_config adapter_ata_ffn_with_cross \
    --gradient_accumulation_steps 2 \
    --encoder_ids_for_language_adapters $encoder_ids_for_lang \
    --encoder_ids_for_task_adapters $encoder_ids_for_task \
    --decoder_ids_for_language_adapters $decoder_ids_for_lang \
    --decoder_ids_for_task_adapters $decoder_ids_for_task \
    --do_online_ewc \
    --importance_for_ewc $importance > $output_dir/training_logs/${seq}.log

sleep 20

rm -rf ../../outputs_online_ewc/peft_with_ewc_${identity}_${epochs}_${importance}_enl_${encoder_ids_for_lang}_ent_${encoder_ids_for_task}_del_${decoder_ids_for_lang}_det_${decoder_ids_for_task}/*/*/checkpoint*

