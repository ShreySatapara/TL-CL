import argparse
import adapters
from adapters import BnConfig
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trainer import PEFTEWCTrainer, FisherInformation
from transformers import DataCollatorForSeq2Seq
import json
import torch
import sys
import os
from inference import eval_peft
from dataset import decaNLPStyleDatasetTorch

import time
import copy

torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from prettytable import PrettyTable



def main():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_checkpoint_path', type=str, default='google/mt5-small')
    # task sequence
    parser.add_argument('--task_sequence', type=str)
    # epochs
    parser.add_argument('--num_train_epochs_per_task', type=int, default=25)
    # per_device_batch_size
    parser.add_argument('--per_device_batch_size', type=int, default=8)
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=1e-4)
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
    parser.add_argument('--total_examples_for_ER', type=int, default=1500)
    # task wise examples for ER
    parser.add_argument('--task_wise_examples_for_ER', type=int, default=None)
    # peft config
    parser.add_argument('--adapter_config', type=str, default=None)
    # gradient accumulation steps
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    # encoder_ids_for_language_adapters
    parser.add_argument('--encoder_ids_for_language_adapters', type=str, default="")
    # decoder_ids_for_language_adapters
    parser.add_argument('--decoder_ids_for_language_adapters', type=str, default="")
    # encoder_ids_for_task_adapters
    parser.add_argument('--encoder_ids_for_task_adapters', type=str, default="")
    # decoder_ids_for_task_adapters
    parser.add_argument('--decoder_ids_for_task_adapters', type=str, default="")
    parser.add_argument('--importance_for_ewc', type=float, default=10)
    parser.add_argument('--cls_nli_en_decoder', action='store_true')
    parser.add_argument('--freeze_language_adapters_after_first_task', action='store_true')
    parser.add_argument('--freeze_task_adapters_after_first_language', action='store_true')
    # do_online_ewc store_true
    parser.add_argument('--do_online_ewc', action='store_true')
    parser.add_argument('--do_zero_shot_inference', action='store_true')
    
    
    args = parser.parse_args()
    
    x = PrettyTable()
    x.field_names = ["PARAMETER", "VALUE"]
    for k,v in vars(args).items():
        x.add_row([k,v])
    print(x)
    
    trained_language_list = []
    trained_task_list = []
    
    # coonvert encoder_ids_for_languge_adapters to list of int
    if args.encoder_ids_for_language_adapters != "":
        args.encoder_ids_for_language_adapters = [int(x) for x in args.encoder_ids_for_language_adapters.split(",")]
    else:
        args.encoder_ids_for_language_adapters = []
    if args.decoder_ids_for_language_adapters != "":
        args.decoder_ids_for_language_adapters = [int(x) for x in args.decoder_ids_for_language_adapters.split(",")]
    else:
        args.decoder_ids_for_language_adapters = []
    if args.encoder_ids_for_task_adapters != "":
        args.encoder_ids_for_task_adapters = [int(x) for x in args.encoder_ids_for_task_adapters.split(",")]
    else:
        args.encoder_ids_for_task_adapters = []
    if args.decoder_ids_for_task_adapters != "":
        args.decoder_ids_for_task_adapters = [int(x) for x in args.decoder_ids_for_task_adapters.split(",")]
    else:
        args.decoder_ids_for_task_adapters = []
    task_list_json = json.load(open(args.task_sequence_file))
    task_list = task_list_json[args.task_sequence].split(",")
    
    
    # get adapter config
    if(args.adapter_config=="adapter_ata_ffn_with_cross"):
        config = BnConfig(reduction_factor=16,
                             non_linearity='gelu',
                             mh_adapter = True,
                             output_adapter = True,
                             cross_adapter = True)
    else:
        raise ValueError("Invalid adapter config")
    
    
    # init model as dummy for trainer class
    #model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
    if args.do_train:
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint_path)
        
        # init fisher_information_class
        fisher_info = FisherInformation()
        
        # for task_lang in task_list:
        for task_lang in task_list:
            task,language = task_lang.split("_")
            #trainer = PEFTEWCTrainer(model = model, args = training_args)
            decoder_language = language
            if args.cls_nli_en_decoder:
                if task=='cls' or task=='nli':
                    decoder_language='en'        
    
    
            # init model
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
            # init adapters
            adapters.init(model)
            # add adapters
            model.add_adapter("cls", config)
            model.set_active_adapters("cls")
            
            # init training args
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
                save_total_limit=0,
                report_to="tensorboard",
                logging_dir=args.root_output_dir,
                overwrite_output_dir=True,
                seed=42,
                data_seed=42,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lr_scheduler_type="constant",
                
            )
            
            if(training_args.local_rank == 0):
                print(f"{'#'*20}\tTASK: {task}\tLANGUAGE: {language}\t{'#'*20}")
            save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, task + "-" + language)
            training_args.output_dir = save_dir_path
            training_args.logging_dir = os.path.join(args.root_output_dir, args.task_sequence,"tensorboard_logs", args.adapter_config, task + "-" + language)  
            
            # get training data
            train_dataset = decaNLPStyleDatasetTorch(args.root_data_dir, 
                                                        task, 
                                                        language, 
                                                        'train', 
                                                        tokenizer,  
                                                        task_sequence=task_list, 
                                                        experience_reply=args.do_experience_reply,
                                                        total_examples_for_ER=args.total_examples_for_ER,
                                                        task_wise_examples_for_ER=args.task_wise_examples_for_ER)
            # get validation data
            val_dataset = decaNLPStyleDatasetTorch(args.root_data_dir, task, language, 'val', tokenizer)
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            
            # lad language specific adapters in encoder
            if os.path.exists(f"{args.root_output_dir}/{args.task_sequence}/{language}/encoder_adapter_state_dict.pth"):
                model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{language}/encoder_adapter_state_dict.pth")
            # load language specific decoder adapters en for cls and nli for all languages
            if os.path.exists(f"{args.root_output_dir}/{args.task_sequence}/{decoder_language}/decoder_adapter_state_dict.pth"):
                model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{decoder_language}/decoder_adapter_state_dict.pth")
                                    
            # load task specific encoder adapters
            if os.path.exists(f"{args.root_output_dir}/{args.task_sequence}/{task}/encoder_adapter_state_dict.pth"):
                model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{task}/encoder_adapter_state_dict.pth")            
            if os.path.exists(f"{args.root_output_dir}/{args.task_sequence}/{task}/decoder_adapter_state_dict.pth"):
                model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{task}/decoder_adapter_state_dict.pth")            
            
            model.train_adapter("cls")
            
            frozen_language_adapters = False
            frozen_task_adapters = False
            
            if(args.freeze_language_adapters_after_first_task):
                if language in trained_language_list:
                    model = freeze_adapters(model, encoder_or_decoder='encoder', layer_id_list = args.encoder_ids_for_language_adapters)
                    model = freeze_adapters(model, encoder_or_decoder='decoder', layer_id_list = args.decoder_ids_for_language_adapters)
                    frozen_language_adapters = True
            if(args.freeze_task_adapters_after_first_language):
                if task in trained_task_list:
                    model = freeze_adapters(model, encoder_or_decoder='encoder', layer_id_list = args.encoder_ids_for_task_adapters)
                    model = freeze_adapters(model, encoder_or_decoder='decoder', layer_id_list = args.decoder_ids_for_task_adapters)    
                    frozen_task_adapters = True
                    
            #if task_lang == task_list[0]:
            trainer = PEFTEWCTrainer(model = model, 
                                    args = training_args,
                                    train_dataset = train_dataset,
                                    eval_dataset = val_dataset,
                                    data_collator = data_collator,
                                    tokenizer = tokenizer,
                                    importance=args.importance_for_ewc,
                                    frozen_language_adapters = frozen_language_adapters,
                                    frozen_task_adapters = frozen_task_adapters,
                                    )

            fisher_info.fisher_information = {task:{},language:{'encoder':{},'decoder':{}}}
            
            trainer.fisher_information = fisher_info.fisher_information
            # load language specific fisher information for encoders
            if os.path.exists(f"{args.root_output_dir}/{args.task_sequence}/{language}/encoder_fisher_information.pth"):
                language_fisher_info = torch.load(f"{args.root_output_dir}/{args.task_sequence}/{language}/encoder_fisher_information.pth")
                trainer.fisher_information[language]['encoder'] = language_fisher_info#.to(trainer.args.device)
                
            if os.path.exists(f"{args.root_output_dir}/{args.task_sequence}/{decoder_language}/decoder_fisher_information.pth"):
                language_fisher_info = torch.load(f"{args.root_output_dir}/{args.task_sequence}/{decoder_language}/decoder_fisher_information.pth")
                trainer.fisher_information[decoder_language]['decoder'] = language_fisher_info#.to(trainer.args.device)
        

            # if task_adapters exist load:
            if os.path.exists(f"{args.root_output_dir}/{args.task_sequence}/{task}/fisher_information.pth"):
                task_fisher_info = torch.load(f"{args.root_output_dir}/{args.task_sequence}/{task}/fisher_information.pth")
                trainer.fisher_information[task] = task_fisher_info#.to(trainer.args.device)
                
            # update trainer.prev_params
            fisher_info.update_previous_params(trainer.model)
            trainer.prev_params = fisher_info.prev_params
            trainer.task = task
            trainer.language = language
            trainer.decoder_language = decoder_language
            # train model()
            trainer.train()
            # save adapters()
            encoder_language_save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, language)
            decoder_language_save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, decoder_language)
            task_save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, task)
            if trainer.args.local_rank == 0:
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path,exist_ok=True)
                trainer.model.save_adapter(save_directory = save_dir_path, adapter_name = "cls")
            
            # save task and language specific adapters seprately
                # save language specific encoder adapters
                #encoder_language_save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, language)
                if not os.path.exists(encoder_language_save_dir_path):
                    os.makedirs(encoder_language_save_dir_path,exist_ok=True)
                    
                save_task_language_adapters(trainer.model,
                                        encoder_language_save_dir_path, 
                                        'encoder',
                                        args.encoder_ids_for_language_adapters, 
                                        args.decoder_ids_for_language_adapters)
                
                # save language specific adapters for decoders
                #decoder_language_save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, decoder_language)
                if not os.path.exists(decoder_language_save_dir_path):
                    os.makedirs(decoder_language_save_dir_path,exist_ok=True)
                save_task_language_adapters(trainer.model,
                                        decoder_language_save_dir_path, 
                                        'decoder',
                                        args.encoder_ids_for_language_adapters, 
                                        args.decoder_ids_for_language_adapters)
            
                # saving task specific adapters for encoders and decoders
                #task_save_dir_path = os.path.join(args.root_output_dir, args.task_sequence, task)
                if not os.path.exists(task_save_dir_path):
                    os.makedirs(task_save_dir_path,exist_ok=True)
                save_task_language_adapters(trainer.model,
                                            task_save_dir_path,
                                            'encoder',
                                            args.encoder_ids_for_task_adapters,
                                            args.decoder_ids_for_task_adapters)
                save_task_language_adapters(trainer.model,
                                            task_save_dir_path,
                                            'decoder',
                                            args.encoder_ids_for_task_adapters,
                                            args.decoder_ids_for_task_adapters)
            # # trainer.update_ewc()
            fisher_info.update_ewc(trainer.model,
                                task,
                            language,
                            decoder_language,
                            args.encoder_ids_for_language_adapters,
                            args.encoder_ids_for_task_adapters,
                            args.decoder_ids_for_language_adapters,
                            args.decoder_ids_for_task_adapters,
                            adapter_name = "cls",
                            dataloader=trainer.get_train_dataloader(),
                            device=trainer.args.device,
                            frozen_language_adapters = frozen_language_adapters,
                            frozen_task_adapters = frozen_task_adapters,
                            online_ewc = args.do_online_ewc,
                            encoder_language_fisher_info_path = encoder_language_save_dir_path + '/encoder_fisher_information.pth',
                            decoder_language_fisher_info_path = decoder_language_save_dir_path + '/decoder_fisher_information.pth',    
                            task_fisher_info_path = task_save_dir_path + '/fisher_information.pth',
            )                            
                            #prev_fisher_info = trainer.fisher_information.clone())  #copy.deepcopy(trainer.fisher_information))
            if training_args.local_rank == 0:
                save_fisher_information(fisher_info.fisher_information[language]['encoder'], encoder_language_save_dir_path + '/encoder_fisher_information.pth')
                save_fisher_information(fisher_info.fisher_information[decoder_language]['decoder'], decoder_language_save_dir_path + '/decoder_fisher_information.pth')
                save_fisher_information(fisher_info.fisher_information[task], task_save_dir_path + '/fisher_information.pth')
            
            trained_language_list.append(language)
            trained_task_list.append(task)
    time.sleep(20)
    ####################### INFERENCE #######################
    if args.do_eval:   
        if args.do_zero_shot_inference:
            task_list = task_list_json['zero_shot_seq'].split(",")
        
        results_dict = {'end_of_task_seq':{args.task_sequence:[]}, 'task_wise':{args.task_sequence:[]}}
        # for task_lang in task_list:
        for task_lang in task_list:
            
            
            task,language = task_lang.split("_")
            decoder_language = language
            if args.cls_nli_en_decoder:
                if task=='cls' or task=='nli':
                    decoder_language='en'
            # init model
            
            # init training args
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
                save_total_limit=0,
                report_to="tensorboard",
                logging_dir=args.root_output_dir,
                overwrite_output_dir=True,
                seed=42,
                data_seed=42,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lr_scheduler_type="constant",
            )
            if(training_args.local_rank == 0):
                print(f"{'#'*20}\tTASK: {task}\tLANGUAGE: {language}\t{'#'*20}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
            # init adapters
            adapters.init(model)
            # add adapters
            model.add_adapter("cls", config)
            model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{language}/encoder_adapter_state_dict.pth")
            model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{decoder_language}/decoder_adapter_state_dict.pth")
            model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{task}/encoder_adapter_state_dict.pth")
            model = load_task_language_adapters(model, f"{args.root_output_dir}/{args.task_sequence}/{task}/decoder_adapter_state_dict.pth")
            model.set_active_adapters("cls")
            
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            # get test data
            test_dataset = decaNLPStyleDatasetTorch(args.root_data_dir, task, language, 'test', tokenizer)
            
            results = eval_peft(model = model,
                                tokenizer = tokenizer,
                                training_args = training_args,
                                data_collator = data_collator,
                                root_output_dir = args.root_output_dir,
                                task = task,
                                language = language,
                                test_dataset = test_dataset)
            results_dict['end_of_task_seq'][args.task_sequence].append(results)
            if not args.do_zero_shot_inference:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_checkpoint_path)
                # init adapters
                adapters.init(model)
                # add adapters
                model.add_adapter("cls", config)
                model.load_adapter(f"{args.root_output_dir}/{args.task_sequence}/{task}-{language}")
                model.set_active_adapters("cls")
                
                results = eval_peft(model = model,
                                    tokenizer = tokenizer,
                                    training_args = training_args,
                                    data_collator = data_collator,
                                    root_output_dir = args.root_output_dir,
                                    task = task,
                                    language = language,
                                    test_dataset = test_dataset)
                
                results_dict['task_wise'][args.task_sequence].append(results)
            
        if not os.path.exists(f"{args.root_results_dir}/{args.task_sequence}"):
            os.makedirs(f"{args.root_results_dir}/{args.task_sequence}",exist_ok=True)
        with open(f"{args.root_results_dir}/{args.task_sequence}/results.json", "w") as f:
            json.dump(results_dict, f)

    
def save_task_language_adapters(model, save_path, encoder_or_decoder=None, encoder_blocks = [0,1,2,3,4,5], decoder_blocks = [0,1,2,3,4,5]):
    
    #print("saveing task and language adapters at ", save_path)
    state_dict = {}
    if encoder_or_decoder == "encoder":
        
        for block in encoder_blocks:
            for i in range(2):
                try:
                    state_dict[f'encoder.block.{block}.layer.{i}.adapters.cls.adapter_down.0.weight'] = model.encoder.block[block].layer[i].adapters.cls.adapter_down[0].weight
                    state_dict[f'encoder.block.{block}.layer.{i}.adapters.cls.adapter_down.0.bias'] = model.encoder.block[block].layer[i].adapters.cls.adapter_down[0].bias
                    state_dict[f'encoder.block.{block}.layer.{i}.adapters.cls.adapter_up.weight'] = model.encoder.block[block].layer[i].adapters.cls.adapter_up.weight
                    state_dict[f'encoder.block.{block}.layer.{i}.adapters.cls.adapter_up.bias'] = model.encoder.block[block].layer[i].adapters.cls.adapter_up.bias
                except:
                    continue
        torch.save(state_dict, save_path + '/encoder_adapter_state_dict.pth')
        
    if encoder_or_decoder == "decoder":
        for block in decoder_blocks:
            for i in range(3):
                try:
                    state_dict[f'decoder.block.{block}.layer.{i}.adapters.cls.adapter_down.0.weight'] = model.decoder.block[block].layer[i].adapters.cls.adapter_down[0].weight
                    state_dict[f'decoder.block.{block}.layer.{i}.adapters.cls.adapter_down.0.bias'] = model.decoder.block[block].layer[i].adapters.cls.adapter_down[0].bias
                    state_dict[f'decoder.block.{block}.layer.{i}.adapters.cls.adapter_up.weight'] = model.decoder.block[block].layer[i].adapters.cls.adapter_up.weight
                    state_dict[f'decoder.block.{block}.layer.{i}.adapters.cls.adapter_up.bias'] = model.decoder.block[block].layer[i].adapters.cls.adapter_up.bias
                except:
                    continue
                
        torch.save(state_dict, save_path + '/decoder_adapter_state_dict.pth')
        
        
def freeze_adapters(model, encoder_or_decoder, layer_id_list):
    if encoder_or_decoder=='encoder':
        for block in layer_id_list:
            for i in range(2):
                try:
                    model.encoder.block[block].layer[i].adapters.cls.adapter_down[0].weight.requires_grad = False
                    model.encoder.block[block].layer[i].adapters.cls.adapter_down[0].bias.requires_grad = False
                    model.encoder.block[block].layer[i].adapters.cls.adapter_up.weight.requires_grad = False
                    model.encoder.block[block].layer[i].adapters.cls.adapter_up.bias.requires_grad = False
                except:
                    continue
    if encoder_or_decoder=='decoder':
        for block in layer_id_list:
            for i in range(3):
                try:
                    model.decoder.block[block].layer[i].adapters.cls.adapter_down[0].weight.requires_grad = False
                    model.decoder.block[block].layer[i].adapters.cls.adapter_down[0].bias.requires_grad = False
                    model.decoder.block[block].layer[i].adapters.cls.adapter_up.weight.requires_grad = False
                    model.decoder.block[block].layer[i].adapters.cls.adapter_up.bias.requires_grad = False
                except:
                    continue
    # print required_grad status
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(n)
    return model

def load_task_language_adapters(model, save_path):
    #print("loading language and task specific adapters from ", save_path)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict, strict=False)
    return model


def save_fisher_information(fisher_info, save_path):
    torch.save(fisher_info, save_path)


if __name__ == '__main__':
    main()