from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import EarlyStoppingCallback, IntervalStrategy
    
import os
import json
from torch.utils.data import Dataset
import numpy as np
import random
import argparse 
import time
from utils import print_argparse_args
from inference import eval

random.seed(42)
import torch
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(42)

class decaNLPStyleDatasetTorchMultitaskMultilingual(Dataset):
    def __init__(self,
                root_dir,  
                split, 
                tokenizer, 
                task_sequence,
            ):
        self.root_dir = root_dir
        self.split = split
        self.tokenizer = tokenizer
        self.task_sequence = task_sequence
        #self.data_path = os.path.join(root_dir, task,lang, split + ".jsonl")
        
        self.data = []
        for item in self.task_sequence:
            task,language = item.split("_")
            data_path = os.path.join(root_dir,task,language, split + ".jsonl")
            with open(data_path) as f:
                for line in f:
                    self.data.append(json.loads(line))
        
        #if(self.split == 'train'):
        random.shuffle(self.data)
        #self.data = self.data[:1000]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Tokenize the texts
        
        question = self.data[idx]["question"]
        context = self.data[idx]["context"]
        answer = self.data[idx]["answer"]
        inputs = self.tokenizer(question, context, truncation=True, max_length=512, padding=True, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs["labels"] = self.tokenizer(answer, truncation=True, max_length=256, padding=True,return_tensors="pt")["input_ids"].squeeze()
        return inputs

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_dir', type=str)
    # model name or path
    parser.add_argument('--model_name_or_path', type=str)
    # root output directory
    parser.add_argument('--root_output_dir', type=str)
    # number of epochs
    parser.add_argument('--num_train_epochs', type=int)
    # per device batch size
    parser.add_argument('--per_device_batch_size', type=int)
    # gradient accumulation steps
    parser.add_argument('--gradient_accumulation_steps', type=int)
    # learning rate
    parser.add_argument('--learning_rate', type=float)
    # results dict
    parser.add_argument('--results_dict', type=str)
    # logging steps
    parser.add_argument('--logging_steps', type=int)
    # do train
    parser.add_argument('--do_train', action='store_true')
    # do eval
    parser.add_argument('--do_eval', action='store_true')
    # task_sequence
    parser.add_argument('--task_sequence', type=str, default="seq1")
    # task_sequence file
    parser.add_argument('--task_sequence_file', type=str, default="task_sequence.json")
    parser.add_argument('--lr_scheduler', type=str, default="constant")
    
    # task list
    #task_list = json.load(open("task_sequence.json"))["task_sequence"].split(',')
        
    args = parser.parse_args()
    
    task_list = json.load(open(args.task_sequence_file))[args.task_sequence].split(',')
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.root_output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        predict_with_generate=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=100,
        save_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
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
        lr_scheduler_type=args.lr_scheduler,
    )
    
    if(training_args.local_rank == 0):
        print_argparse_args(parser)
        
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_dataset = decaNLPStyleDatasetTorchMultitaskMultilingual(root_dir = args.root_data_dir,
                                                            split = "train",
                                                            tokenizer=tokenizer,
                                                            task_sequence = task_list)
    
    val_dataset = decaNLPStyleDatasetTorchMultitaskMultilingual(root_dir = args.root_data_dir,
                                                            split = "val",
                                                            tokenizer=tokenizer,
                                                            task_sequence = task_list)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    if(args.do_train):
        
        
        trainer.train()
        trainer.save_model(args.root_output_dir + "/best_checkpoint")
        time.sleep(20)   
    if(args.do_eval):
        answer_len_dict = {
            "cls":256,
            "nli":256,
            "qa":256,
            "summ":256   
        }
        #task_list = args.task_list.split(',')
        if(training_args.local_rank == 0):
            #print_argparse_args(parser)
            print(f"{'#'*20}\tINFERENCE\t{'#'*20}")
            
        results_dict = []
        #for train_item in task_list:
        train_task, train_language = task_list[-1].split('_')
            
        if(training_args.local_rank == 0):
            print(f"{'#'*20}\tTASK: {train_task}\tLANGUAGE: {train_language}\t{'#'*20}")
        
        training_model_path = args.root_output_dir + "/best_checkpoint" #os.path.join(args.root_output_dir, args.task_sequence, train_task + "-" + train_language)
        tokenizer = AutoTokenizer.from_pretrained(training_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(training_model_path)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        results = eval(model = model, 
                    task_sequence = task_list,#args.task_sequence, 
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
        
        # of os path not exist
        if not os.path.exists(args.results_dict):
            os.makedirs(args.results_dict)
        with open(os.path.join(args.results_dict, args.task_sequence + ".json"), 'w') as f:
            json.dump(results_dict, f)
            
            
if __name__ == "__main__":
    main()