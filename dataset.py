import os
import json
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

random.seed(42)
import torch
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(42)

class decaNLPStyleDatasetTorch(Dataset):
    def __init__(self,
                root_dir, 
                task, 
                lang, 
                split, 
                tokenizer, 
                answer_len_dict = None, 
                experience_reply = False, 
                task_sequence = None, 
                total_examples_for_ER = None,
                task_wise_examples_for_ER = None,    
            ):
        self.root_dir = root_dir
        self.task = task
        self.lang = lang
        self.split = split
        self.tokenizer = tokenizer
        #self.answer_len = answer_len_dict[task]
        self.data_path = os.path.join(root_dir, task,lang, split + ".jsonl")
        
        # read jsonl from data_path
        self.data = []
        with open(self.data_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        
        #print("data len", len(self.data))
        if(experience_reply):
            if(task_wise_examples_for_ER is not None or total_examples_for_ER is not None):
                self.data.extend(self.get_data_for_ER(task, lang, task_sequence, total_examples_for_ER, task_wise_examples_for_ER))
            else:
                raise Exception("Please provide either task_wise_examples_for_ER or total_examples_for_ER")        
        #print("data len after ER: ", len(self.data))
        if(self.split == 'train'):
            random.shuffle(self.data)
        print("Data length: ", len(self.data))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Tokenize the texts
        
        question = self.data[idx]["question"]
        context = self.data[idx]["context"]
        answer = self.data[idx]["answer"]
        
        # print(answer)
        # print when answer is empty
        # if(answer == ""):
        #     print(self.split, self.lang, question)
        #     print(self.data[idx])
        # # print when answer is nan
        # if(type(answer) == float):
        #     print(self.split, self.lang, question)
        #     print(self.data[idx])
        
        inputs = self.tokenizer(question, context, truncation=True, max_length=512, padding=True, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs["labels"] = self.tokenizer(answer, padding="longest",return_tensors="pt")["input_ids"].squeeze()
        return inputs
    
    def get_examples_from_task(self, task, language, split, no_of_examples = None):
        data_path = os.path.join(self.root_dir, task,language, split + ".jsonl")
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        # shuffle 
        random.seed(42)
        random.shuffle(data)
        #print(data[0])
        return data[:no_of_examples]

    def get_data_for_ER(self, task, language, task_sequence, total_examples_for_ER, task_wise_examples_for_ER):
        
        task_len = f"{task}_{language}"

        task_sequence_np = np.array(task_sequence)
        task_sequence_index = np.where(task_sequence_np == task_len)[0][0]
        
        #print(task_sequence_index)
        
        er_data = []
        if task_sequence_index == 0:
        
            #print("returning empty list")
        
            return er_data
        
        #task_wise_examples_for_ER = 80 #total_examples_for_ER // task_sequence_index
        
        # if(task_wise_examples_for_ER is not None):
        #    if(total_examples_for_ER is not None):
        #        task_wise_examples_for_ER = total_examples_for_ER // task_sequence_index
        
        for item in task_sequence[:task_sequence_index]:
            task, language = item.split('_')
            data = self.get_examples_from_task(task, language, self.split, task_wise_examples_for_ER)
            er_data.extend(data)
        
        # print("len of er data",len(er_data))
        return er_data