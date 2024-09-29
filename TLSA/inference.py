from transformers import Seq2SeqTrainer
from dataset import decaNLPStyleDatasetTorch
from metrics import compute_score
from utils import print_result_table
from adapters import Seq2SeqAdapterTrainer
import adapters

def eval_individual_task(trainer, task, dataset):
    predictions = trainer.predict(dataset)
    decoded_preds = []
    predictions.predictions[predictions.predictions == -100] = trainer.tokenizer.pad_token_id
    predictions.label_ids[predictions.label_ids == -100] = trainer.tokenizer.pad_token_id
    decoded_preds = trainer.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = trainer.tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    score = compute_score(task, decoded_preds, decoded_labels)
    return score

def eval_peft(model, tokenizer, training_args, data_collator, root_output_dir, task, language,test_dataset):
    
    results = []
       
    trainer = Seq2SeqAdapterTrainer(
    model = model,
    args = training_args,
    tokenizer = tokenizer, 
    data_collator = data_collator,
    )
    
    #test_dataset = decaNLPStyleDatasetTorch(root_data_dir, task, language, 'test', tokenizer, answer_len_dict = answer_len_dict)
    
    score = eval_individual_task(trainer, task, test_dataset)
    
    results.append({
        "task": task,
        "language": language,
        "score": score
    })
    
    if(training_args.local_rank == 0):
        print_result_table(results)
    return results