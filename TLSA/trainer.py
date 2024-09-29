from adapters import Seq2SeqAdapterTrainer
import adapters
import os
# import Union
from typing import Union, Optional, Dict, List, Tuple, Callable
import torch
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from torch import nn
from datasets import Dataset
from tqdm import tqdm

class PEFTEWCTrainer(Seq2SeqAdapterTrainer):
    def __init__(self, model, importance, frozen_language_adapters, frozen_task_adapters, *args, **kwargs):
        
        super().__init__(model, *args, **kwargs)
        self.importance = importance
        self.fisher_information = {}
        self.prev_params = {}
        self.task = None
        self.language = None
        self.decoder_language = None
        self.frozen_language_adapters = frozen_language_adapters
        self.frozen_task_adapters = frozen_task_adapters
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        task_ewc = 0
        language_encoder_ewc = 0
        language_decoder_ewc = 0
        
        if(model.training):
            model = model.module
        
        if not self.frozen_task_adapters:
            if self.fisher_information.get(self.task,None):
                for n,p in model.named_parameters():
                    if n in self.fisher_information[self.task] and p.requires_grad:
                        task_ewc += (self.fisher_information[self.task][n].to(self.args.device) * (p - self.prev_params[n]) ** 2).sum()
        
        if not self.frozen_task_adapters:
            if self.fisher_information.get(self.language,None):
                if self.fisher_information[self.language].get('encoder',None):
                    for n,p in model.named_parameters():
                        if n in self.fisher_information[self.language]['encoder'] and p.requires_grad:
                            language_encoder_ewc += (self.fisher_information[self.language]['encoder'][n].to(self.args.device) * (p - self.prev_params[n]) ** 2).sum()
            
            if self.fisher_information.get(self.decoder_language,None):
                if self.fisher_information[self.decoder_language].get('decoder',None):
                    for n,p in model.named_parameters():
                        if n in self.fisher_information[self.decoder_language]['decoder'] and p.requires_grad:
                            language_decoder_ewc += (self.fisher_information[self.decoder_language]['decoder'][n].to(self.args.device) * (p - self.prev_params[n]) ** 2).sum()
        
        loss += self.importance * (task_ewc + language_encoder_ewc + language_decoder_ewc)
        return (loss, outputs) if return_outputs else loss
        
class FisherInformation:
    def __init__(self):
        self.fisher_information = {}
        self.prev_params = {}
        
    def update_ewc(self,
                   model,
                   task,
                   language,
                   decoder_language,
                   encoder_ids_for_language_adapters,
                   encoder_ids_for_task_adapters,
                   decoder_ids_for_language_adapters,
                   decoder_ids_for_task_adapters,
                   adapter_name,
                   dataloader,
                   device,
                   frozen_language_adapters,
                   frozen_task_adapters,
                   online_ewc=False,
                   encoder_language_fisher_info_path=None,
                   decoder_language_fisher_info_path=None,
                   task_fisher_info_path=None,):
                   #prev_fisher_info=None,):
        #dataloader = self.get_train_dataloader()
        task_specific_params = []
        language_specific_params = []
        self.fisher_information[language] = {}
        self.fisher_information[decoder_language] = {}
        self.fisher_information[language]['encoder'] = {}
        self.fisher_information[decoder_language]['decoder'] = {}
        self.fisher_information[task] = {}
        if not frozen_language_adapters:
            for block in encoder_ids_for_language_adapters:
                for i in range(2):
                    try:
                        self.fisher_information[language]['encoder'][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.weight'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].weight)
                        self.fisher_information[language]['encoder'][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.bias'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].bias)
                        self.fisher_information[language]['encoder'][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.weight'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_up.weight)
                        self.fisher_information[language]['encoder'][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.bias'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_up.bias)
                    except:
                        continue
        if not frozen_task_adapters:
            for block in encoder_ids_for_task_adapters:
                for i in range(2):
                    try:
                        self.fisher_information[task][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.weight'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].weight)
                        self.fisher_information[task][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.bias'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].bias)
                        self.fisher_information[task][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.weight'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_up.weight)
                        self.fisher_information[task][f'encoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.bias'] = torch.zeros_like(model.encoder.block[block].layer[i].adapters[adapter_name].adapter_up.bias)
                    except:
                        continue    
        
        if not frozen_language_adapters:
            for block in decoder_ids_for_language_adapters:
                for i in range(3):
                    try:
                        self.fisher_information[decoder_language]['decoder'][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.weight'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].weight)
                        self.fisher_information[decoder_language]['decoder'][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.bias'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].bias)
                        self.fisher_information[decoder_language]['decoder'][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.weight'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_up.weight)
                        self.fisher_information[decoder_language]['decoder'][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.bias'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_up.bias)
                    except:
                        continue
        if not frozen_task_adapters:
            for block in decoder_ids_for_task_adapters:
                for i in range(3):
                    try:
                        self.fisher_information[task][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.weight'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].weight)
                        self.fisher_information[task][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_down.0.bias'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_down[0].bias)
                        self.fisher_information[task][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.weight'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_up.weight)
                        self.fisher_information[task][f'decoder.block.{block}.layer.{i}.adapters.{adapter_name}.adapter_up.bias'] = torch.zeros_like(model.decoder.block[block].layer[i].adapters[adapter_name].adapter_up.bias)
                    except:
                        continue
        
        model.eval()
        for batch in tqdm(dataloader, desc="Computing Fisher Information", unit="batch",total=len(dataloader)):
            batch = batch.to(device)
            model.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            for n,p in model.named_parameters():
                if n in self.fisher_information[language]['encoder']:
                    self.fisher_information[language]['encoder'][n] += p.grad.data ** 2
                if n in self.fisher_information[decoder_language]['decoder']:
                    self.fisher_information[decoder_language]['decoder'][n] += p.grad.data ** 2
                if n in self.fisher_information[task]:
                    self.fisher_information[task][n] += p.grad.data ** 2
            
        for n,p in model.named_parameters():
            if n in self.fisher_information[language]['encoder']:
                self.fisher_information[language]['encoder'][n] /= len(dataloader)
            if n in self.fisher_information[decoder_language]['decoder']:
                self.fisher_information[decoder_language]['decoder'][n] /= len(dataloader)
            if n in self.fisher_information[task]:
                self.fisher_information[task][n] /= len(dataloader)
        if online_ewc: #and prev_fisher_info:
            language_encoder_flag = False
            language_decoder_flag = False
            task_flag = False
            prev_fisher_info = {}
            prev_fisher_info[language] = {} 
            prev_fisher_info[decoder_language] = {}
            prev_fisher_info[task] = {}
            # calcualte averge of self.fisher_information and prev_fisher_info
            
            
            # for n,p in model.named_parameters():
            #     if n in self.fisher_information[language]['encoder'] and n in prev_fisher_info[language]['encoder']:
            #         self.fisher_information[language]['encoder'][n] = (self.fisher_information[language]['encoder'][n] + prev_fisher_info[language]['encoder'][n]) / 2
            #         print("im in")
            #     if n in self.fisher_information[decoder_language]['decoder'] and n in prev_fisher_info[decoder_language]['decoder']:
            #         print("im in decoder")
            #         self.fisher_information[decoder_language]['decoder'][n] = (self.fisher_information[decoder_language]['decoder'][n] + prev_fisher_info[decoder_language]['decoder'][n]) / 2
            #     if n in self.fisher_information[task] and n in prev_fisher_info[task]:
            #         self.fisher_information[task][n] = (self.fisher_information[task][n] + prev_fisher_info[task][n]) / 2
            #         print("im in task")
            
            # load language_fisher_info
            if encoder_language_fisher_info_path:
                if os.path.exists(encoder_language_fisher_info_path):
                    prev_fisher_info[language]['encoder'] = torch.load(encoder_language_fisher_info_path,map_location=model.device)#.to(model.device)
                    language_encoder_flag = True
            if decoder_language_fisher_info_path:
                if os.path.exists(decoder_language_fisher_info_path):
                    prev_fisher_info[decoder_language]['decoder'] = torch.load(decoder_language_fisher_info_path,map_location=model.device)#.to(model.device)
                    language_decoder_flag = True
            # load task_fisher_info
            if task_fisher_info_path:
                if os.path.exists(task_fisher_info_path):
                    prev_fisher_info[task] = torch.load(task_fisher_info_path,map_location=model.device)#.to(model.device)
                    task_flag = True
            for n,p in model.named_parameters():
                # average the fisher information
                if language_encoder_flag and n in self.fisher_information[language]['encoder']:
                    self.fisher_information[language]['encoder'][n] = (self.fisher_information[language]['encoder'][n] + prev_fisher_info[language]['encoder'][n])/2#.to(self.fisher_information[language]['encoder'][n].device)) / 2
                    print("im in")
                if language_decoder_flag and n in self.fisher_information[decoder_language]['decoder']:
                    self.fisher_information[decoder_language]['decoder'][n] = (self.fisher_information[decoder_language]['decoder'][n] + prev_fisher_info[decoder_language]['decoder'][n])/2#.to(self.fisher_information[decoder_language]['decoder'][n].device)) / 2
                    print("im in decoder")
                if task_flag and n in self.fisher_information[task]:
                    self.fisher_information[task][n] = (self.fisher_information[task][n] + prev_fisher_info[task][n]) / 2
                    print("im in task")
            print("online ewc successfully done")
            del prev_fisher_info
            torch.cuda.empty_cache()
        model.train()
    
    def update_previous_params(self,model):
        #print('im here')
        self.prev_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    