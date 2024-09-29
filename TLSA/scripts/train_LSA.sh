conda activate new_transformers_torch

gpu_id=$1
port_no=$2
seq_no=$3

identity=language_only_adapters


epochs=25
encoder_ids_for_lang=0,1,2,3,4,5,6,7
decoder_ids_for_lang=0,1,2,3,4,5,6,7
encoder_ids_for_task=""
decoder_ids_for_task=""

importance=1e3

output_dir=../../outputs/peft_with_ewc_${identity}_${epochs}_${importance}_enl_${encoder_ids_for_lang}_ent_${encoder_ids_for_task}_del_${decoder_ids_for_lang}_det_${decoder_ids_for_task}
mkdir -p $output_dir/training_logs

CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch --main_process_port $port_no main.py \
    --model_name_or_checkpoint_path google/mt5-small \
    --task_sequence $seq_no \
    --num_train_epochs_per_task $epochs \
    --per_device_batch_size 8 \
    --learning_rate 1e-4 \
    --root_output_dir $output_dir \
    --root_results_dir ../../results/peft_with_ewc_${identity}_${epochs}_${importance}_enl_${encoder_ids_for_lang}_ent_${encoder_ids_for_task}_del_${decoder_ids_for_lang}_det_${decoder_ids_for_task} \
    --root_data_dir ../../data \
    --logging_steps 100 \
    --do_eval \
    --do_train \
    --task_sequence_file ../../task_sequences.json \
    --adapter_config adapter_ata_ffn_with_cross \
    --gradient_accumulation_steps 2 \
    --encoder_ids_for_language_adapters $encoder_ids_for_lang \
    --decoder_ids_for_language_adapters $decoder_ids_for_lang \
    --importance_for_ewc $importance 2>&1 | tee $output_dir/training_logs/$seq_no.log

sleep 20

rm -rf ../../outputs/peft_with_ewc_${identity}_${epochs}_${importance}_enl_${encoder_ids_for_lang}_ent_${encoder_ids_for_task}_del_${decoder_ids_for_lang}_det_${decoder_ids_for_task}/*/*/checkpoint*
