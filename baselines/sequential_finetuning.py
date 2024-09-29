from transformers import Seq2SeqTrainer
from transformers import EarlyStoppingCallback

class SequentialFineTuner:
    def train(self, model, tokenizer, training_args, train_dataset, val_dataset, data_collator, save_dir_path):
        seq_trainer = Seq2SeqTrainer(
            model = model,
            args=training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            data_collator = data_collator,
            tokenizer = tokenizer,
        )
        
        seq_trainer.train()
        seq_trainer.save_model(save_dir_path)
        return seq_trainer.model