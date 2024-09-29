from adapters import Seq2SeqAdapterTrainer
import adapters
import os
from adapters import BnConfig
from adapters.composition import Stack
from adapters import AdapterConfig
from adapters import SeqBnInvConfig
from adapters import SeqBnConfig
from adapters.composition import Stack
# create adapter config with seqbn for task adapters and seq_bn config with invadapter true for langugae adapter



lang_adapter_config = SeqBnInvConfig(reduction_factor=16)
task_adapter_config = SeqBnConfig(reduction_factor=16)

class MADXCL:
    def train(self,
              model, 
              tokenizer, 
              training_args,
              train_dataset,
              val_dataset,
              data_collator,
              save_dir_path,
              task,
              language,):
        print(save_dir_path)
        
        adapters.init(model)
        if not os.path.exists(f"{save_dir_path}/{language}"):
            model.add_adapter(f"{language}", lang_adapter_config)
        else:
            model.load_adapter(f"{save_dir_path}/{language}", lang_adapter_config)
        
        if not os.path.exists(f"{save_dir_path}/{task}"):
            model.add_adapter(f"{task}", task_adapter_config)
        else:
            model.load_adapter(f"{save_dir_path}/{task}", task_adapter_config)
        # model.train_adapter(f"{language}")
        # model.train_adapter(f"{task}")
        #model.active_adapters = Stack(f"{language}", f"{task}")
        model.set_active_adapters(Stack(f"{language}", f"{task}"))
        model.train_adapter(Stack(f"{language}", f"{task}"))
        
        # model.train_adapter(f"{language}")
        # model.train_adapter(f"{task}")

        
        # print no of trainble and frozen params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if training_args.local_rank == 0:
            print(model.adapter_summary())
            print(f"Trainable params: {trainable_params}")
            print(f"Frozen params: {frozen_params}")
        
        adapter_trainer = Seq2SeqAdapterTrainer(
            model = model,
            args=training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            data_collator = data_collator,
            tokenizer = tokenizer,
        )
            
        adapter_trainer.train()
        # save langugae adapter
        adapter_trainer.model.save_adapter(f"{save_dir_path}/{language}", adapter_name=f"{language}")
        # save task adapter
        adapter_trainer.model.save_adapter(f"{save_dir_path}/{task}", adapter_name=f"{task}")
        # save task-language pair adapter        
        if not os.path.exists(f"{save_dir_path}/{task}-{language}"):
            os.makedirs(f"{save_dir_path}/{task}-{language}",exist_ok=True)
        adapter_trainer.model.save_adapter(f"{save_dir_path}/{task}-{language}/{language}", adapter_name=f"{language}")
        adapter_trainer.model.save_adapter(f"{save_dir_path}/{task}-{language}/{task}", adapter_name=f"{task}")


