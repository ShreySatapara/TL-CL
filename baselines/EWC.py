import torch
from transformers import Seq2SeqTrainer  #TrainingArguments

class EwcTrainer(Seq2SeqTrainer):
    def __init__(self, model, fisher_matrix=None, importance=10, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.fisher_matrix = fisher_matrix
        self.importance = importance
        self.prev_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad} 

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # Apply EWC penalty
        ewc_loss = 0
        if(model.training):
            model = model.module
        if self.fisher_matrix:
            #print("fisher matrix: \n\n ",self.fisher_matrix.keys())
            #print("prev params: \n\n ",self.prev_params.keys())
            for n, p in model.named_parameters():
                #print("n: ",n)
                if p.requires_grad:
                    loss += (self.fisher_matrix[n] * (p - self.prev_params[n]) ** 2).sum() * self.importance
                    ewc_loss += (self.fisher_matrix[n] * (p - self.prev_params[n]) ** 2).sum() * self.importance
        #if(self.args.local_rank == 0):
        #    print("ewc loss: ", ewc_loss, "CE loss:", outputs.loss.item())
        #print("ewc loss: ", ewc_loss.item(), "CE loss:", outputs.loss.item())
        return (loss, outputs) if return_outputs else loss

    def update_ewc(self, dataloader, importance=10):
        # Calculate Fisher Information matrix and update previous parameters
        self.fisher_matrix = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher_matrix[n] = torch.zeros_like(p)

        self.model.eval()
        for batch in dataloader:
            batch = batch.to(self.args.device)
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.fisher_matrix[n] += p.grad ** 2

        for n in self.fisher_matrix:
            self.fisher_matrix[n] /= len(dataloader)

        self.importance = importance
        return self.fisher_matrix


class EWCFineTuner:
    def __init__(self):
        self.fisher_matrix = None
    
    def train(self, model, tokenizer, training_args, train_dataset, val_dataset, data_collator, save_dir_path, do_ewc=False):
        #print("in ewc train")
        
        seq_trainer = EwcTrainer(
            model = model,
            args=training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            data_collator = data_collator,
            tokenizer = tokenizer,
            fisher_matrix = self.fisher_matrix
        )
        
        seq_trainer.train()
            #print("do ewc in ewc trainer")
        self.fisher_matrix = seq_trainer.update_ewc(seq_trainer.get_train_dataloader())
        
        seq_trainer.save_model(save_dir_path)
        
        return seq_trainer.model