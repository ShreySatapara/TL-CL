import collections
from rouge import Rouge
import re
import string
import numpy as np


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, ground_truth):
    #print(prediction, ground_truth)
    return prediction == ground_truth

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    ground_truths = [ground_truths]
    #print(ground_truths)
    for idx, ground_truth in enumerate(ground_truths):
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def computeEM(outputs, targets):
    normalized_outputs = [normalize_text(o) for o in outputs]
    normalized_targets = [normalize_text(t) for t in targets]
    outs = [metric_max_over_ground_truths(exact_match, o, t) for o, t in zip(normalized_outputs, normalized_targets)]
    return sum(outs) / len(outputs) * 100

def f1_score(prediction, ground_truth):
    prediction_tokens =  prediction.split()
    ground_truth_tokens =  ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def computeF1(outputs, targets):
    return sum([metric_max_over_ground_truths(f1_score, o, t) for o, t in zip(outputs, targets)]) / len(outputs) * 100

def computeRouge(predictions, ground_truths):
    rouge = Rouge()
    rouge_scores = {
        'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
    }
    for i in range(len(predictions)):
        predicted_text = predictions[i]
        reference_text = ground_truths[i]

        # Skip empty strings for either predicted or reference text
        
        #if predicted_text.strip() and reference_text.strip():
            # Handle empty strings, for example, by assigning a low ROUGE score
            # You can adjust this logic as needed
        try:
            rouge_score = rouge.get_scores(predicted_text, reference_text)[0]
        except:
            rouge_score = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
            # Calculate ROUGE scores using the 'rouge' library
            
        rouge_scores['rouge-1']['f'] += rouge_score['rouge-1']['f']
        rouge_scores['rouge-1']['p'] += rouge_score['rouge-1']['p']
        rouge_scores['rouge-1']['r'] += rouge_score['rouge-1']['r']
        rouge_scores['rouge-2']['f'] += rouge_score['rouge-2']['f']
        rouge_scores['rouge-2']['p'] += rouge_score['rouge-2']['p']
        rouge_scores['rouge-2']['r'] += rouge_score['rouge-2']['r']
        rouge_scores['rouge-l']['f'] += rouge_score['rouge-l']['f']
        rouge_scores['rouge-l']['p'] += rouge_score['rouge-l']['p']
        rouge_scores['rouge-l']['r'] += rouge_score['rouge-l']['r']
    rouge_scores['rouge-1']['f'] /= len(predictions)
    rouge_scores['rouge-1']['p'] /= len(predictions)
    rouge_scores['rouge-1']['r'] /= len(predictions)
    rouge_scores['rouge-2']['f'] /= len(predictions)
    rouge_scores['rouge-2']['p'] /= len(predictions)
    rouge_scores['rouge-2']['r'] /= len(predictions)
    rouge_scores['rouge-l']['f'] /= len(predictions)
    rouge_scores['rouge-l']['p'] /= len(predictions)
    rouge_scores['rouge-l']['r'] /= len(predictions)
    #scores = rouge.get_scores(predictions, ground_truths, avg=True)
    return ((rouge_scores['rouge-1']['f'] + rouge_scores['rouge-2']['f'] + rouge_scores['rouge-l']['f'])/3 )* 100


def compute_score(task, predictions, ground_truths):
    if(task=="cls" or task=="nli"):
        return {"metrics":"EM","score":computeEM(predictions, ground_truths)}
    elif(task=="qa"):
        return {"metrics":"F1","score":computeF1(predictions, ground_truths)}
    elif(task=="summ"):
        return {"metrics":"Rouge","score":computeRouge(predictions, ground_truths)}
    else:
        raise Exception("Invalid task")