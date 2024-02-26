import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import string
from simcse import SimCSE
import torch
import re
import numpy as np
import jsonlines
import argparse
from tqdm import tqdm, trange
from ipdb import set_trace
from utils import *
import sys

DEBUG = True
simcse_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")


# parser = argparse.ArgumentParser()
# parser.add_argument("--filename", type=str, help="Input Filename", required=True)
# parser.add_argument("--threshold", default=0.75, type=float, help="Threshold of similarity")
# parser.add_argument("--model", default="princeton-nlp/sup-simcse-roberta-large", type=str, help="Smilarity Model")
# args = parser.parse_args()

def eval(filename):
    # Load the data
    with open(filename, 'r') as f:
        data_list = json.load(f)  # Load the file and parse the JSON array
    threshold = 0.7
    # model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_length = len(data_list)

    # Initialize counter for correct matches
    correct_matches = 0


    # Iterate through the data
    for i in trange(data_length):
        meaning = data_list[i]['meaning']
        prediction = data_list[i]['prediction']

        # Get the similarity score
        similarity = simcse_model.similarity(meaning, prediction, device=device)
        if DEBUG:
            print(f'Similarity: {similarity:.3f}')

        # Update correct matches counter based on threshold
        if similarity >= threshold:
            correct_matches += 1

    # Calculate accuracy
    accuracy = correct_matches / data_length
    print(f'Correct matches: {correct_matches} in {data_length} examples')
    print(f'Accuracy: {accuracy:.4f}')


def calculate_similarity_and_judge(str1, str2, threshold=0.7):
    """
    Calculate the similarity between two strings using SimCSE and judge if they are similar.

    Parameters:
    str1: The first string
    str2: The second string
    threshold: The threshold for judging similarity, default is 0.7

    Returns:
    similarity_score: The similarity score between the two strings
    is_similar: A boolean indicating whether the strings are considered similar
    """
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    similarity_score = simcse_model.similarity(str1, str2, device=device)
    is_similar = similarity_score >= threshold
    return similarity_score, is_similar

# # Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# str1 = "This is a sentence."
# str2 = "This is another sentence."

# similarity_score, is_similar = calculate_similarity_and_judge(model, str1, str2, device=device)
# print(f"Similarity Score: {similarity_score:.3f}, Is Similar: {is_similar}")



if __name__ == '__main__':
    eval(sys.argv[1])