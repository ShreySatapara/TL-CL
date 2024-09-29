gpu_ids=$1
port_no=$2
seq=$3


encoder_ids_for_lang=0,2
encoder_ids_for_task=1,3,4,5,6,7
decoder_ids_for_lang=5,7
decoder_ids_for_task=0,1,2,3,4,6
importance=1e3
epochs=25


identifire=peft_with_ewc_${epochs}_${importance}_enl_${encoder_ids_for_lang}_ent_${encoder_ids_for_task}_del_${decoder_ids_for_lang}_det_${decoder_ids_for_task}_freeze_language_adapters_after_first_time_training

output_dir=../../outputs/$identifire
mkdir -p $output_dir/training_logs



CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch --main_process_port $port_no main.py \
    --model_name_or_checkpoint_path google/mt5-small \
    --task_sequence $seq \
    --num_train_epochs_per_task $epochs \
    --per_device_batch_size 8 \
    --learning_rate 1e-4 \
    --root_output_dir $output_dir \
    --root_results_dir ../../results/$identifire \
    --root_data_dir ../../data \
    --logging_steps 100 \
    --do_eval \
    --task_sequence_file ../../task_sequences.json \
    --adapter_config adapter_ata_ffn_with_cross \
    --gradient_accumulation_steps 2 \
    --encoder_ids_for_language_adapters $encoder_ids_for_lang \
    --encoder_ids_for_task_adapters $encoder_ids_for_task \
    --decoder_ids_for_language_adapters $decoder_ids_for_lang \
    --decoder_ids_for_task_adapters $decoder_ids_for_task \
    --importance_for_ewc $importance \
    --freeze_language_adapters_after_first_task > $output_dir/training_logs/${seq}.log

sleep 20

rm -rf ../../outputs/$identifire/*/*/checkpoint*

sleep 20
