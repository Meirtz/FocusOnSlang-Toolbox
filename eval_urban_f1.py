import sys
import ujson as json
import re
import string
from collections import Counter
import nltk
from nltk.translate import bleu_score
import pickle
from eval_urban_simcse import calculate_similarity_and_judge
import evaluate
from sentence_transformers import SentenceTransformer
ss_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
import torch

from sklearn.metrics.pairwise import cosine_similarity

# Ensure you have the NLTK punkt tokenizer downloaded
nltk.download('punkt')

def normalize_answer(s):
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

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge_score_func(prediction, ground_truth):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=[prediction], references=[ground_truth])
    return results

def sentence_similarity(sentence1, sentence2):
    embeddings1 = ss_model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = ss_model.encode(sentence2, convert_to_tensor=True)

    # Normalize the embeddings to unit vectors
    embeddings1 = embeddings1 / embeddings1.norm(dim=0)
    embeddings2 = embeddings2 / embeddings2.norm(dim=0)

    # Compute cosine similarity as dot product
    similarity = torch.mm(embeddings1.unsqueeze(0), embeddings2.unsqueeze(0).transpose(0, 1))
    
    return similarity.item()  # Convert to a regular number

def bleu_score_func(prediction, ground_truth):
    prediction_tokens = nltk.word_tokenize(prediction)
    ground_truth_tokens = [nltk.word_tokenize(ground_truth)]
    return bleu_score.sentence_bleu(
        ground_truth_tokens, 
        prediction_tokens, 
        weights = (1/2, 1/3, 1/6, 0)
        )


def batch_sentence_similarity(sentences1, sentences2):
    embeddings1 = ss_model.encode(sentences1, convert_to_tensor=True, batch_size=32)
    embeddings2 = ss_model.encode(sentences2, convert_to_tensor=True, batch_size=32)

    embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
    embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)

    similarity_matrix = torch.mm(embeddings1, embeddings2.transpose(0, 1))
    return torch.diag(similarity_matrix).cpu().tolist()

def batch_rouge_score_func(predictions, golds):
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=predictions, references=golds, use_aggregator=False)

    # Initialize batch scores
    batch_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}

    # Accumulate the scores
    for score in scores['rouge1']:
        batch_scores['rouge-1'].append(score)
    for score in scores['rouge2']:
        batch_scores['rouge-2'].append(score)
    for score in scores['rougeL']:
        batch_scores['rouge-l'].append(score)

    # Calculate average for each metric
    avg_scores = {k: sum(v) / len(v) if v else 0 for k, v in batch_scores.items()}
    return avg_scores


def eval(prediction_file, batch_size=256):
    metrics = {
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'bleu': 0, 
        'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'sentence_similarity': 0,
        'simcse_score': 0, 'simcse_judgement': 0 
    }
    N = 0

    with open(prediction_file, 'r') as f:
        data = json.load(f)

    # Process in batches
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        predictions = [item.get('prediction', '') for item in batch_data]
        golds = [item.get('meaning', '') for item in batch_data]

        # Batch computation for sentence similarity
        # batch process preprocess_prediction_and_ground_truth
        processed_predictions = []
        processed_golds = []
        for prediction, gold in zip(predictions, golds):
            # print(prediction, gold)
            processed_prediction, processed_ground_truth = preprocess_prediction_and_ground_truth(prediction, gold)
            print(processed_prediction, processed_ground_truth)
            # print(a)
            processed_predictions.append(processed_prediction)
            processed_golds.append(processed_ground_truth)
        
        # sentence_similarities = batch_sentence_similarity(predictions, golds)
        sentence_similarities = batch_sentence_similarity(processed_predictions, processed_golds)

        # Batch computation for ROUGE scores
        batch_rouge_scores = batch_rouge_score_func(predictions, golds)

        for j, item in enumerate(batch_data):
            prediction = item.get('prediction', '')
            gold = item.get('meaning', '')

            # prediction, gold = preprocess_prediction_and_ground_truth(prediction, gold)

            em = exact_match_score(prediction, gold)
            f1, prec, recall = f1_score(prediction, gold)
            bleu = bleu_score_func(prediction, gold)
            # rouge = rouge_score_func(prediction, gold)
            sentence_sim = sentence_similarities[j]

            simcse_score, simcse_judgement = calculate_similarity_and_judge(prediction, gold)

            # Accumulate metrics
            metrics['em'] += float(em)
            metrics['f1'] += f1
            metrics['prec'] += prec
            metrics['recall'] += recall
            metrics['bleu'] += bleu
            metrics['sentence_similarity'] += sentence_sim
            metrics['rouge-1'] += batch_rouge_scores['rouge-1']
            metrics['rouge-2'] += batch_rouge_scores['rouge-2']
            metrics['rouge-l'] += batch_rouge_scores['rouge-l']
            metrics['simcse_score'] += simcse_score
            metrics['simcse_judgement'] += simcse_judgement
            N += 1

    # Average the metrics
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)



def preprocess_prediction_and_ground_truth(prediction, ground_truth):
    # Define the regex patterns for different parts of the expressions
    patterns = {
        "refers_to": r"(.+?) refers to (.+?)\.",
        "often_used": r"It is often used to (.+?)\.",
        "expression": r"This expression (.+)"
    }

    def extract_parts(text):
        # Extracts the specified parts from the text using regex patterns
        extracted_parts = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match and len(match.groups()) >= 2:
                extracted_parts[key] = match.group(2)
            else:
                extracted_parts[key] = None
        return extracted_parts

    # Extract parts from both prediction and ground truth
    prediction_parts = extract_parts(prediction)
    ground_truth_parts = extract_parts(ground_truth)

    # Combine the extracted parts into a single string for each input
    combined_prediction = ', '.join(filter(None, prediction_parts.values()))
    combined_ground_truth = ', '.join(filter(None, ground_truth_parts.values()))

    return combined_prediction, combined_ground_truth

# Example usage
prediction = "Megathon refers to a set of very hard exercise sessions lead by a fitness instructor named Megan Davies. It's often used when talking about a difficult workout challenge that happens over several days. This expression gives an idea of how hard the workouts are and how they're meant to make someone stronger, help their heart and breathing, and burn fat."
ground_truth = "Megathon refers to an intense fitness program by Megan Davies. It is often used in the context of rigorous multi-day exercise challenges. This expression conveys the intensity and health benefits of the workouts."

processed_prediction, processed_ground_truth = preprocess_prediction_and_ground_truth(prediction, ground_truth)


# Example usage
prediction = "Megathon refers to a set of very hard exercise sessions lead by a fitness instructor named Megan Davies. It's often used when talking about a difficult workout challenge that happens over several days. This expression gives an idea of how hard the workouts are and how they're meant to make someone stronger, help their heart and breathing, and burn fat."
ground_truth = "Megathon refers to an intense fitness program by Megan Davies. It's often used in the context of rigorous multi-day exercise challenges. This expression conveys the intensity and health benefits of the workouts."

processed_prediction, processed_ground_truth = preprocess_prediction_and_ground_truth(prediction, ground_truth)

    


def calculate_metrics(prediction, ground_truth):

    

    # Calculate exact match score
    em = exact_match_score(prediction, ground_truth)

    # Calculate F1, precision, and recall
    f1, prec, recall = f1_score(prediction, ground_truth)

    # Calculate BLEU score
    bleu = bleu_score_func(prediction, ground_truth)

    # Calculate SimCSE score

    simcse_score, simcse_judgement = calculate_similarity_and_judge(prediction, ground_truth)

    # Compile the results into a dictionary
    results = {
        'exact_match': em,
        'f1_score': f1,
        'precision': prec,
        'recall': recall,
        'bleu_score': bleu,
        'simcse_score': simcse_score,
        'simcse_judgement': simcse_judgement
    }

    # Calculate ROUGE score
    rouge_scores = rouge_score_func(prediction, ground_truth)

    # Add ROUGE scores to the results dictionary
    results['rouge'] = rouge_scores

    # Calculate sentence similarity
    sim_score = sentence_similarity(prediction, ground_truth)

    # Add sentence similarity to the results dictionary
    results['sentence_similarity'] = sim_score

    return results


if __name__ == '__main__':
    eval(sys.argv[1])
