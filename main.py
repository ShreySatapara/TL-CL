from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from baselines.sequential_finetuning import SequentialFineTuner
from baselines.EWC import EWCFineTuner
from baselines.parameter_isolation import ParameterIsolation
from baselines.parameter_isolation import *
from baselines.MAD_X import *


import torch.distributed as dist
from transformers import IntervalStrategy, EarlyStoppingCallback
from adapters import AutoAdapterModel
from dataset import decaNLPStyleDatasetTorch
from utils import *
from inference import eval, eval_peft, eval_mad_x

import argparse
import os
import json
import warnings
import time
# set seed
import torch
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(42)
import random
random.seed(42)




warnings.filterwarnings("ignore")

# The line `from prettytable import PrettyTable` is importing the `PrettyTable` class from the
# `prettytable` module. This class is used to create and display formatted tables in a text-based
# interface.
from prettytable import PrettyTable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_checkpoint_path', type=str, default='t5-small')
    # training method
    parser.add_argument('--training_method', type=str, default='sequential_finetuning')    
    # task sequence
    parser.add_argument('--task_sequence', type=str)
    # epochs
    parser.add_argument('--num_train_epochs_per_task', type=int, default=10)
    # per_device_batch_size
    parser.add_argument('--per_device_batch_size', type=int, default=8)
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    # root output dir
    parser.add_argument('--root_output_dir', type=str, default='output')
    # root data dir
    parser.add_argument('--root_data_dir', type=str, default='data')
    # root results dir
    parser.add_argument('--root_results_dir', type=str, default='results')
    # logging steps
    parser.add_argument('--logging_steps', type=int, default=100)
    # do train store true
    parser.add_argument('--do_train', action='store_true')
    # do eval store true
    parser.add_argument('--do_eval', action='store_true')
    # task sequence file
    parser.add_argument('--task_sequence_file', type=str, default='task_sequence.json')
    # do experimence reply
    parser.add_argument('--do_experience_reply', action='store_true')
    # total examples for ER
    parser.add_argument('--total_examples_for_ER', type=int, default=None)
    # task wise examples for ER
    parser.add_argument('--task_wise_examples_for_ER', type=int, default=None)
    # peft config
    parser.add_argument('--peft_config', type=str, default=None)
    # gradient accumulation steps
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    # lr_schedular
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    # final_checkpoint_path
    parser.add_argument('--final_checkpoint_path', type=str, default=None)
    # sequence_to_infer
    parser.add_argument('--sequence_to_infer', type=str, default=None)
    # do_zero_shot_inference store_true
    parser.add_argument('--do_zero_shot_inference', action='store_true')
    parser.add_argument('--do_peft_with_er', action='store_true')
    # args pretrain lang adapter store_ture
    
    
    args = parser.parse_args()
    
    answer_len_dict = {
        "cls":256,
        "nli":256,
        "qa":256,
        "summ":256
    }

    task_sequences = json.load(open(args.task_sequence_file))    
    task_list = task_sequences[args.task_sequence].split(',')
    
    if(args.training_method == 'sequential_finetuning'):
        trainer = SequentialFineTuner()
        
    elif(args.training_method == 'ewc'):
        #print("EWC")
        trainer = EWCFineTuner()
    elif(args.training_method == 'peft'):
        trainer = ParameterIsolation()
    # elif(args.training_method == 'peft_with_ewc'):
    #     trainer = PEFTEWC(importance=10)
    elif(args.training_method == 'mad_x_cl'):
        trainer = MADXCL()
    #task_seq_path = args.task_list.replace(',','-')
    if(args.do_train):
            
        # path not exist
        if not os.path.exists(args.root_output_dir):
            os.makedirs(args.root_output_dir)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
        if args.do_peft_with_er:
            adapter_config = args.peft_config
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
            model.add_adapter(f"adapter", config)
            model.set_active_adapters(f"adapter")
            model.train_adapter(f"adapter")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint_path)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        for item in task_list:
            
            training_args = Seq2SeqTrainingArguments(
                output_dir=args.root_output_dir,
                do_train=True,
                do_eval=True,
                do_predict=True,
                predict_with_generate=True,
                evaluation_strategy=IntervalStrategy.STEPS,
                save_strategy="steps",
                eval_steps=100,
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.per_device_batch_size,
                per_device_eval_batch_size=args.per_device_batch_size,
                num_train_epochs=args.num_train_epochs_per_task,
                logging_strategy="steps",
                logging_steps=args.logging_steps,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                save_total_limit=0,
                report_to="tensorboard",
                logging_dir=args.root_output_dir,
                overwrite_output_dir=True,
                seed=42,
                data_seed=42,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lr_scheduler_type=args.lr_scheduler,
            )
            
            if(training_args.local_rank == 0):
                print_argparse_args(parser)
            
            task, language = item.split('_')
            if(training_args.local_rank == 0):
                print(f"{'#'*20}\tTASK: {task}\tLANGUAGE: {language}\t{'#'*20}")
            
            if args.training_method == 'mad_x_cl':
                save_dir_path = os.path.join(args.root_output_dir, args.task_sequence)
            else:
                save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, task + "-" + language)
            
            training_args.output_dir = save_dir_path
            if(args.training_method=="peft"):
                training_args.logging_dir = os.path.join(args.root_output_dir, "tensorboard_logs", args.peft_config, task + "-" + language)
            elif(args.training_method=="mad_x_cl"):
                training_args.logging_dir = os.path.join(args.root_output_dir, "tensorboard_logs", task + "-" + language)
            else:
                training_args.logging_dir = save_dir_path + "/logs"
            
            # train_dataset = decaNLPStyleDataset(args.root_data_dir, task, language, 'val', tokenizer, answer_len_dict = answer_len_dict)
            # val_dataset = decaNLPStyleDataset(args.root_data_dir, task, language, 'val', tokenizer,answer_len_dict = answer_len_dict)
            train_dataset = decaNLPStyleDatasetTorch(args.root_data_dir, 
                                                    task, 
                                                    language, 
                                                    'train', 
                                                    tokenizer, 
                                                    answer_len_dict = answer_len_dict, 
                                                    task_sequence=task_list, 
                                                    experience_reply=args.do_experience_reply,
                                                    total_examples_for_ER=args.total_examples_for_ER,
                                                    task_wise_examples_for_ER=args.task_wise_examples_for_ER)
            val_dataset = decaNLPStyleDatasetTorch(args.root_data_dir, task, language, 'val', tokenizer,answer_len_dict = answer_len_dict)
            if(args.training_method=="sequential_finetuning"):
                model = trainer.train(
                            model=model, 
                            tokenizer=tokenizer, 
                            training_args=training_args, 
                            train_dataset=train_dataset,#.get_dataset(), 
                            val_dataset=val_dataset,#.get_dataset(), 
                            data_collator=data_collator,
                            save_dir_path = save_dir_path
                            )
            elif(args.training_method=="ewc"):
                #print('ewc finetuning')
                task_len = f"{task}_{language}"
                task_list_np = np.array(task_list)
                task_list_index = np.where(task_list_np == task_len)[0][0]
                if(task_list_index)>0: 
                    do_ewc = True
                else:
                    do_ewc = False
                #print("task index and do ewc: ",task_list_index, do_ewc)
                model = trainer.train(
                            model=model, 
                            tokenizer=tokenizer, 
                            training_args=training_args, 
                            train_dataset=train_dataset,#.get_dataset(), 
                            val_dataset=val_dataset,#.get_dataset(), 
                            data_collator=data_collator,
                            save_dir_path = save_dir_path,
                            do_ewc=do_ewc
                            )
        
            elif(args.training_method=="peft"):
                #model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
                training_args.output_dir = f"{args.root_output_dir}/{args.peft_config}" 
                trainer.train(
                    model = model,
                    tokenizer = tokenizer,
                    training_args = training_args,
                    train_dataset = train_dataset,
                    val_dataset = val_dataset,
                    data_collator = data_collator,
                    save_dir_path = args.root_output_dir,
                    adapter_config = args.peft_config,
                    task = task,
                    language = language  
                )
            elif(args.training_method=="mad_x_cl"):
                # training_args.output_dir = f"{args.root_output_dir}"
                trainer.train(
                    model = model,
                    tokenizer = tokenizer,
                    training_args = training_args,
                    train_dataset = train_dataset,
                    val_dataset = val_dataset,
                    data_collator = data_collator,
                    save_dir_path = save_dir_path,#args.root_output_dir,
                    task = task,
                    language = language,)
                
            
    time.sleep(20)
    if(args.do_eval):
        training_args = Seq2SeqTrainingArguments(
                output_dir=args.root_output_dir,
                do_train=True,
                do_eval=True,
                do_predict=True,
                predict_with_generate=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.per_device_batch_size,
                per_device_eval_batch_size=args.per_device_batch_size,
                num_train_epochs=args.num_train_epochs_per_task,
                logging_strategy="steps",
                logging_steps=args.logging_steps,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                save_total_limit=1,
                report_to="tensorboard",
                logging_dir=args.root_output_dir,
                overwrite_output_dir=True,
                seed=42,
                data_seed=42,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lr_scheduler_type="constant",
            )
        
        if(training_args.local_rank == 0):
            #print_argparse_args(parser)
            print(f"{'#'*20}\tINFERENCE\t{'#'*20}")
        
        results_dict = []
        if args.do_zero_shot_inference:
            
            if args.do_peft_with_er:    
                model = AutoAdapterModel.from_pretrained(args.final_checkpoint_path)
                model.set_active_adapters(f"adapter")
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.final_checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(args.final_checkpoint_path)
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            # print(task_sequences)
            # print(args.sequence_to_infer)
            # print(task_sequences[args.sequence_to_infer])
            seq_to_infer = task_sequences[args.sequence_to_infer].split(',')
            # print(seq_to_infer)
            for task_lang in seq_to_infer:
                # print(task_lang)
                results = eval(model = model,
                               task_sequence = [task_lang], #args.task_sequence,
                            root_data_dir = args.root_data_dir,
                            tokenizer = tokenizer, 
                            answer_len_dict = answer_len_dict,
                            training_args=training_args,
                            data_collator=data_collator
                            )
                results_dict.append({
                    "seq_id":args.task_sequence,
                    "task":task_lang.split('_')[0],
                    "language":task_lang.split('_')[1],
                    "results":results            
                })
            if not os.path.exists(args.root_results_dir):
                os.makedirs(args.root_results_dir, exist_ok=True)
            with open(os.path.join(args.root_results_dir, args.task_sequence + f"_{args.sequence_to_infer}.json"), 'w') as f:
                json.dump(results_dict, f)
            return
        if(args.training_method=="sequential_finetuning" or args.training_method=="ewc"):  
            for train_item in task_list:    
                train_task, train_language = train_item.split('_')
                
                if(training_args.local_rank == 0):
                    print(f"{'#'*20}\tTASK: {train_task}\tLANGUAGE: {train_language}\t{'#'*20}")
                
                training_model_path = os.path.join(args.root_output_dir, args.task_sequence, train_task + "-" + train_language)
                tokenizer = AutoTokenizer.from_pretrained(training_model_path)
                
                if args.do_peft_with_er:
                    # model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
                    # adapter_config = args.peft_config
                    # if(adapter_config == "adapter_ffn"):
                    #     config = adapter_ffn
                    # elif(adapter_config == "adapter_ata_ffn"):
                    #     config = adapter_ata_ffn
                    # elif(adapter_config == "adapter_ata_ffn_with_cross"):
                    #     config = adapter_ata_ffn_with_cross
                    # elif(adapter_config == "pfx_config"):
                    #     config = pfx_config
                    # elif(adapter_config == "pfx_config_cross"):
                    #     config = pfx_config_cross
                    # elif(adapter_config == "adapter_ata_ffn_with_cross_pfx"):
                    #     config = adapter_ata_ffn_with_cross_pfx
                    # else:
                    #     raise ValueError("Invalid adapter config")
                    
                    # adapters.init(model)
                    # model.add_adapter(f"adapter", config)
                    # model.set_active_adapters(f"adapter")
                    # import safetensors
                    # from safetensors import safe_open
                    # tensors = {}
                    # with safe_open(training_model_path + "/model.safetensors", framework="pt", device="cpu") as f:
                    #     for key in f.keys():
                    #         tensors[key] = f.get_tensor(key)
                    # model.load_state_dict(tensors)
                    
                    model = AutoAdapterModel.from_pretrained(training_model_path)
                    model.set_active_adapters(f"adapter")
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(training_model_path)
                # load model from model.safetensors
                
                
                data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
                results = eval(model = model, 
                            task_sequence = task_list, #args.task_sequence,
                            root_data_dir = args.root_data_dir,
                            tokenizer = tokenizer, 
                            answer_len_dict = answer_len_dict,
                            training_args=training_args,
                            data_collator=data_collator
                            )
                
                results_dict.append({
                    "seq_id":args.task_sequence,
                    "task":train_task,
                    "language":train_language,
                    "results":results            
                })
            if not os.path.exists(args.root_results_dir):
                os.makedirs(args.root_results_dir, exist_ok=True)
        
            with open(os.path.join(args.root_results_dir, args.task_sequence + ".json"), 'w') as f:
                json.dump(results_dict, f)
        elif(args.training_method=="peft"):
            if(training_args.local_rank == 0):
                print(f"{'#'*20}\tPEFT INFERENCE\t{'#'*20}")
            for train_item in task_list:
                train_task, train_language = train_item.split('_')
                #training_model_path = os.path.join(args.root_output_dir, args.task_sequence, train_task + "-" + train_language)
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
                data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
                results = eval_peft(model = model, 
                            root_data_dir = args.root_data_dir,
                            tokenizer = tokenizer, 
                            answer_len_dict = answer_len_dict,
                            training_args=training_args,
                            data_collator=data_collator,
                            root_output_dir=args.root_output_dir,
                            adapter_config=args.peft_config,
                            task = train_task,
                            language = train_language
                            )
                
                results_dict.append({
                    "seq_id":args.task_sequence,
                    "task":train_task,
                    "language":train_language,
                    "results":results            
                })
            
            if not os.path.exists(f"{args.root_results_dir}/{args.peft_config}"):
                os.makedirs(f"{args.root_results_dir}/{args.peft_config}", exist_ok=True)
            with open(os.path.join(args.root_results_dir, args.peft_config, args.task_sequence + ".json"), 'w') as f:
                json.dump(results_dict, f)
        
        elif args.training_method == "mad_x_cl":


            if(training_args.local_rank == 0):
                print(f"{'#'*20}\tMAD-X INFERENCE\t{'#'*20}")
            for train_item in task_list:
                train_task, train_language = train_item.split('_')
                #training_model_path = os.path.join(args.root_output_dir, args.task_sequence, train_task + "-" + train_language)
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
                data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
                
                save_dir_path = os.path.join(args.root_output_dir, args.task_sequence)
                results = eval_mad_x(model = model, 
                            root_data_dir = args.root_data_dir,
                            tokenizer = tokenizer, 
                            answer_len_dict = answer_len_dict,
                            training_args=training_args,
                            data_collator=data_collator,
                            save_dir_path = save_dir_path,
                            task = train_task,
                            language = train_language
                            )
                
                results_dict.append({
                    "seq_id":args.task_sequence,
                    "task":train_task,
                    "language":train_language,
                    "results":results            
                })
            if not os.path.exists(args.root_results_dir):
                os.makedirs(args.root_results_dir, exist_ok=True)
        
            with open(os.path.join(args.root_results_dir, args.task_sequence + ".json"), 'w') as f:
                json.dump(results_dict, f)
        else:
            raise ValueError("Invalid training method")
        
        # of os path not exist
        

# def load_adapters(model, layer_list, cross_self_both='self', save_path):
#     adapter_state_dict = torch.load(save_path)
    
#     # iterate over list of layers load weights
    
#     return model

# def save_adapters(model, layer_list, cross_self_both='self', save_path):
#     # iterate over list of layers save weights
#     return model




if __name__ == '__main__':
    main()