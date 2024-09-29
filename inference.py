from transformers import Seq2SeqTrainer
from dataset import decaNLPStyleDatasetTorch
from metrics import compute_score
from utils import print_result_table
from adapters import Seq2SeqAdapterTrainer
import adapters
def eval(model, task_sequence, root_data_dir, tokenizer, answer_len_dict, training_args, data_collator):
    
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        tokenizer = tokenizer, 
        data_collator = data_collator,
    )
    results = []
    for item in task_sequence:
        #print(item)
        task, language = item.split('_')
        test_dataset = decaNLPStyleDatasetTorch(root_data_dir, task, language, 'test', tokenizer, answer_len_dict = answer_len_dict)
        # predictions = trainer.predict(test_dataset)
        
        # decoded_preds = []
        # # print(predictions.predictions.shape)
        # # print(predictions.label_ids.shape)
        # # print(predictions)
        # # replace -100 with pad token id
        # predictions.predictions[predictions.predictions == -100] = tokenizer.pad_token_id
        # predictions.label_ids[predictions.label_ids == -100] = tokenizer.pad_token_id
        # #predictions.predictions = predictions.predictions[:,:]
        # # for pred in predictions.predictions:
        # #     decoded_preds.append(tokenizer.decode(pred, skip_special_tokens=True))
        # # decoded_labels = []
        # # for label in predictions.label_ids:
        # #     decoded_labels.append(tokenizer.decode(label, skip_special_tokens=True))
                
        # decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
        # decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
        
        # score = compute_score(task, decoded_preds, decoded_labels)
        score = eval_individual_task(trainer, task, test_dataset)
        
        results.append({
            "task": task,
            "language": language,
            "score": score
        })
    if(training_args.local_rank == 0):
        print_result_table(results)
    return results

def eval_individual_task(trainer, task, dataset):
    predictions = trainer.predict(dataset)
    decoded_preds = []
    predictions.predictions[predictions.predictions == -100] = trainer.tokenizer.pad_token_id
    predictions.label_ids[predictions.label_ids == -100] = trainer.tokenizer.pad_token_id
    decoded_preds = trainer.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = trainer.tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    score = compute_score(task, decoded_preds, decoded_labels)
    return score

    # decoded_preds = []
    # decoded_labels = []
    # for pred in predictions.predictions:
    #     decoded_preds.append(trainer.tokenizer.decode(pred, skip_special_tokens=True))
    # for label in predictions.label_ids:
    #     decoded_labels.append(trainer.tokenizer.decode(label, skip_special_tokens=True))
    # score = compute_score(task, decoded_preds, decoded_labels)
    # return score
    
def eval_peft(model, root_data_dir, tokenizer, 
              answer_len_dict, training_args, data_collator, 
              root_output_dir, adapter_config, task, language):
    
    results = []
    
    #task, language = item.split('_')
    load_dir = f"{root_output_dir}/{adapter_config}/{task}-{language}"
    adapters.init(model)
    adapter_name = model.load_adapter(load_dir)
    model.set_active_adapters(adapter_name)   
    
    trainer = Seq2SeqAdapterTrainer(
    model = model,
    args = training_args,
    tokenizer = tokenizer, 
    data_collator = data_collator,
    )
    
    test_dataset = decaNLPStyleDatasetTorch(root_data_dir, task, language, 'test', tokenizer, answer_len_dict = answer_len_dict)
    
    score = eval_individual_task(trainer, task, test_dataset)
    
    results.append({
        "task": task,
        "language": language,
        "score": score
    })
    
    if(training_args.local_rank == 0):
        print_result_table(results)
    return results


def eval_mad_x(model,
               root_data_dir,
               tokenizer,
               answer_len_dict,
               training_args,
               data_collator,
               save_dir_path,task, language):
    results = []
    adapters.init(model)
    lang_adapter_config = adapters.SeqBnInvConfig(reduction_factor=16)
    task_adapter_config = adapters.SeqBnConfig(reduction_factor=16)
    model.load_adapter(f"{save_dir_path}/{language}", lang_adapter_config)
    model.load_adapter(f"{save_dir_path}/{task}", task_adapter_config)
    model.active_adapters = adapters.Stack(f"{language}", f"{task}")
    trainer = Seq2SeqAdapterTrainer(
        model = model,
        args = training_args,
        tokenizer = tokenizer, 
        data_collator = data_collator,
    )
    
    test_dataset = decaNLPStyleDatasetTorch(root_data_dir, task, language, 'test', tokenizer, answer_len_dict = answer_len_dict)
    score = eval_individual_task(trainer, task, test_dataset)
    results.append({
        "task": task,
        "language": language,
        "score": score
    })
    if(training_args.local_rank == 0):
        print_result_table(results)
    return results