from adapters import Seq2SeqAdapterTrainer
import adapters
import os
from adapters import BnConfig, PrefixTuningConfig, LoRAConfig, ConfigUnion


adapter_ffn = BnConfig(reduction_factor=16,
                       non_linearity='gelu',
                        mh_adapter = False,
                        output_adapter = True,)

adapter_ata_ffn = BnConfig(reduction_factor=16,
                            non_linearity='gelu',
                            mh_adapter = True,
                            output_adapter = True,)

adapter_ata_ffn_with_cross = BnConfig(reduction_factor=16,
                            non_linearity='gelu',
                            mh_adapter = True,
                            output_adapter = True,
                            cross_adapter = True)

pfx_config = PrefixTuningConfig(encoder_prefix=True, prefix_length=30)


adapter_ata_ffn_with_cross_pfx = ConfigUnion(adapter_ata_ffn_with_cross, pfx_config)





class ParameterIsolation:
    
    # load adapter
    # set activw adapter
    # train adapter
    
    def train(self, model, tokenizer, training_args, train_dataset, val_dataset, data_collator, 
              save_dir_path, adapter_config, task, language):
        
        if(adapter_config == "adapter_ffn"):
            config = adapter_ffn
        elif(adapter_config == "adapter_ata_ffn"):
            config = adapter_ata_ffn
        elif(adapter_config == "adapter_ata_ffn_with_cross"):
            config = adapter_ata_ffn_with_cross
        elif(adapter_config == "pfx_config"):
            config = pfx_config
        elif(adapter_config == "pfx_config_cross"):
            config = pfx_config_cross
        elif(adapter_config == "adapter_ata_ffn_with_cross_pfx"):
            config = adapter_ata_ffn_with_cross_pfx
        else:
            raise ValueError("Invalid adapter config")
    
        
        adapters.init(model)
        model.add_adapter(f"{task}-{language}", config)
        model.set_active_adapters(f"{task}-{language}")
        model.train_adapter(f"{task}-{language}")
        
        adapter_trainer = Seq2SeqAdapterTrainer(
            model = model,
            args=training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            data_collator = data_collator,
            tokenizer = tokenizer,
        )
        #print(save_dir_path)
        save_dir_path += f"/{adapter_config}/{task}-{language}"
        #print(save_dir_path)
        
        adapter_trainer.train()
        # if os path not exists save_dir_path
        # create the path
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path,exist_ok=True)
            
        adapter_trainer.model.save_adapter(save_directory = save_dir_path, adapter_name = f"{task}-{language}")