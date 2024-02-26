from llm import *
import prompt
from utils import *
import re
from tqdm import tqdm, trange
import string

def count_valid_words(your_string):
    cleaned_string = re.sub(r'[^\w\s]', '', your_string)
    tokens = cleaned_string.split()
    valid_word_count = len(tokens)
    return valid_word_count


def parse_decision(text):
    """
    Parse the decision from the provided text.

    Args:
    text (str): The text containing the decision line.

    Returns:
    bool: True if the decision is to retain, False if the decision is to discard.
    """
    # Split the text into lines
    lines = text.split('\n')

    # Iterate through each line
    for line in lines:
        # Check if the line contains the decision marker
        if "Decision:" in line:
            # Check the decision and return the appropriate boolean value
            return "Retain" in line

    # Default return value if no decision line is found
    return False


def parse_knowledge_check_decision(text):
    """
    Parse the decision from the provided text.

    Args:
    text (str): The text containing the decision line.

    Returns:
    bool: True if the decision is to retain, False if the decision is to discard.
    """
    # Split the text into lines
    lines = text.split('\n')

    # Iterate through each line
    for line in lines:
        # Check if the line contains the decision marker
        if "Decision:" in line:
            # Check the decision and return the appropriate boolean value
            return "Retain" in line

    # Default return value if no decision line is found
    return False

def parse_transformed_counterfactual(text):
    """
    Parse the analysis and the transformed phrase from the provided text, removing all punctuation at the end of the phrase.

    Args:
    text (str): The text containing the transformed data.

    Returns:
    dict: A dictionary containing the analysis and the transformed phrase with end punctuation removed.
    """
    # Append a log file with text
    with open('./data/transformed_explanation_log.txt', 'a') as f:
        f.write(text)
        f.write('\n\n')

    # Find and extract the "Analysis" part
    analysis_start = text.find("Analysis:")
    analysis_end = text.find("Transformed Phrase:", analysis_start)
    analysis = text[analysis_start:analysis_end].strip() if analysis_start != -1 and analysis_end != -1 else ""

    # Find and extract the "Transformed Phrase" part
    phrase_start = text.find("Transformed Phrase:", analysis_end)
    phrase_end = text.find("\n", phrase_start)
    transformed_phrase = text[phrase_start:phrase_end].strip() if phrase_start != -1 and phrase_end != -1 else ""

    # Clean up the extracted "Transformed Phrase"
    if transformed_phrase.startswith("Transformed Phrase:"):
        transformed_phrase = transformed_phrase.replace("Transformed Phrase:", "").strip()
    if transformed_phrase.startswith('"') and transformed_phrase.endswith('"'):
        transformed_phrase = transformed_phrase[1:-1].strip()

    # Remove all punctuation from the end of the transformed phrase
    while transformed_phrase and transformed_phrase[-1] in string.punctuation:
        transformed_phrase = transformed_phrase[:-1].strip()

    return {"analysis": analysis, "word": transformed_phrase}



def parse_multi_transformed_explanation(text):
    """
    Parse multiple new transformed explanations from the provided text.

    Args:
    text (str): The text containing the new transformed explanations.

    Returns:
    list: A list of strings, each string being a new transformed explanation.
    """
    # Append a log file with text
    with open('./data/transformed_explanation_log.txt', 'a') as f:
        f.write(text)
        f.write('\n\n')

    keywords = [
        "New Transformed Explanation 1:", 
        "New Transformed Explanation 2:", 
        "New Transformed Explanation 3:", 
        "New Transformed Explanation 4:"
    ]

    explanations = []
    last_index = 0

    for keyword in keywords:
        index = text.find(keyword, last_index)
        if index != -1:
            # Capture the start of the next section for slicing
            next_index = len(text)
            for next_keyword in keywords:
                if text.find(next_keyword, index + len(keyword)) != -1:
                    next_index = text.find(next_keyword, index + len(keyword))
                    break

            explanation = text[index + len(keyword):next_index].strip()
            explanations.append(explanation)
            last_index = next_index
        else:
            break  # Keyword not found, exit loop

    return explanations



def check_knowledge_and_save(input_filename, output_filename):
    data_list = read_json(input_filename)
    llm = LargeLanguageModel(use_openai=True, model_name="gpt-4-1106-preview")
    knowledge_checked_data = []

    for item in tqdm(data_list):
        word = item['word']
        example = item['example']
        ref_meaning = item['meaning']


        # Prepare the prompt for LLM to check knowledge
        prompt_text = prompt.model_knowledge_check.format(word=word, meaning=ref_meaning, example=example)


        # Get response from LLM
        response = llm.get_response(prompt_text)
        print(response)

        # Analyze the response
        knowledge_checked_data.append({
            'word': word,
            'meaning': ref_meaning,
            'example': example,
            'known_by_model': 'False' if parse_decision(response) else 'True',
            'Explanation': response.split('Explanation: ')[-1].split('\n')[0]
        })

        print(f'Current cost: ${LargeLanguageModel.calculate_total_cost():.6f}')

    save_dataset(knowledge_checked_data, output_filename)

    


def clean_data(filename, cleaned_filename, sample_num):
    data_list = read_json(filename)


    llm = LargeLanguageModel(use_openai=True, model_name="gpt-3.5-turbo-1106")
    cleaned_data = []
    for i in trange(sample_num):
        word = data_list[i]['word']
        example = data_list[i]['example']
        ref_meaning = data_list[i]['meaning']

        print(f'Word: {word}')
        print(f'Example: {example}')
        print(f'Ref meaning: {ref_meaning}')

        if count_valid_words(example) < 10 or count_valid_words(ref_meaning) < 10:
            continue

        # if word.lower() not in example.lower():
        #     continue

        # Prepare the prompt for LLM
        prompt_text = prompt.clean_data.format(word=word, meaning=ref_meaning, example=example)

        # Get response from LLM
        response = llm.get_response(prompt_text)
        print(f'Current cost: ${LargeLanguageModel.calculate_total_cost():.6f}')

        print(response)

        # Check the final decision from LLM
        if parse_decision(response) is True:
            cleaned_data.append(data_list[i])

    # Save the cleaned data to a new JSON file
    save_dataset(cleaned_data, cleaned_filename)




def transform_data(input_filename, output_filename):
    data_list = read_json(input_filename)
    llm = LargeLanguageModel(use_openai=True, model_name="gpt-4-1106-preview")
    transformed_data = []

    for item in tqdm(data_list):
        word = item['word']
        example = item['example']
        ref_meaning = item['meaning']

        # Prepare the prompt for LLM
        prompt_text = prompt.transform_dict_data.format(word=word, meaning=ref_meaning, example=example)

        # Get response from LLM
        response = llm.get_response(prompt_text)
        print(response)
        # print(f'Current cost: ${LLM.gpt_usage():.7f}')

        # Parse and transform the response
        # Assuming the response includes a 'Transformed Explanation' and 'Transformed Example'

        try:
            result = parse_transformed_counterfactual(response)
            
        except Exception as e:

            print(f'Failed to parse the response for word: {word}')
            print(response)
            print('Failed with error: ', e)

        generate_more_prompt = prompt.generate_new_meaning.format(word=result['word'],
                                                                  transformed_meaning=result['meaning'])
        
        # more meanings
        more_meanings_response = llm.get_response(generate_more_prompt)
        print(more_meanings_response)
        try:
            more_meanings = parse_multi_transformed_explanation(more_meanings_response)
            
        except Exception as e:
            print(f'Failed to parse the response for word: {word}')
            print(more_meanings_response)
            print('Failed with error: ', e)
        
        more_meanings.append(result['meaning'])
        print(f'more_meanings: {more_meanings}')
        # ------------------------------

        result['more_meanings'] = more_meanings
        result['raw_example'] = example
        result['raw_meaning'] = ref_meaning

        transformed_data.append(result)

    save_dataset(transformed_data, output_filename)


import new_prompt

def case_insensitive_replace(text, old, new):
    """
    Replace all occurrences of 'old' with 'new' in 'text', ignoring case and punctuation.

    Args:
    text (str): The original text.
    old (str): The substring to be replaced.
    new (str): The substring to replace with.

    Returns:
    str: The text with all occurrences of 'old' replaced by 'new'.
    """
    # Remove punctuation from 'old' for more flexible matching
    old_clean = re.sub(r'[^\w\s]', '', old)
    # Create a regular expression for case-insensitive matching of 'old'
    regex = re.compile(re.escape(old_clean), re.IGNORECASE)
    # Replace all occurrences of 'old' with 'new'
    return regex.sub(new, text)


def counterfactual_transform(input_filename, output_filename):
    
    data_list = read_json(input_filename)
    llm = LargeLanguageModel(use_openai=True, model_name="gpt-4-1106-preview")
    transformed_data = []

    for item in tqdm(data_list):
        word = item['word']
        example = item['example']
        ref_meaning = item['meaning']
        more_meanings = item['more_meanings']

        # Prepare the prompt for LLM
        prompt_text = new_prompt.counterfactual_transform.format(word=word, meaning=ref_meaning, example=example)

        # Get response from LLM
        response = llm.get_response(prompt_text)
        print(response)
        # print(f'Current cost: ${LLM.gpt_usage():.7f}')

        # Parse and transform the response
        # Assuming the response includes a 'Transformed Explanation' and 'Transformed Example'

        try:
            result = parse_transformed_counterfactual(response)
            
        except Exception as e:

            print(f'Failed to parse the response for word: {word}')
            print(response)
            print('Failed with error: ', e)

        # generate_more_prompt = prompt.generate_new_meaning.format(word=result['word'],
        #                                                           transformed_meaning=ref_meaning.replace(word, result['word']))
            
        
        
        # more meanings
        # more_meanings_response = llm.get_response(generate_more_prompt)
        # print(more_meanings_response)
        # try:
        #     more_meanings = parse_multi_transformed_explanation(more_meanings_response)
            
        # except Exception as e:
        #     print(f'Failed to parse the response for word: {word}')
        #     print(more_meanings_response)
        #     print('Failed with error: ', e)
        try:
            result['meaning'] = case_insensitive_replace(ref_meaning, word, result['word'])
            result['more_meanings'] = []
            for m in more_meanings:
                result['more_meanings'].append(case_insensitive_replace(m, word, result['word']))
            result['example'] = case_insensitive_replace(example, word, result['word'])
        except Exception as e:
            print(f'Failed to replace the word: {word}')
            print('Failed with error: ', e)
            result['meaning'] = ref_meaning
            result['more_meanings'] = more_meanings
            result['example'] = example


        
        # more_meanings.append(result['meaning'])
        # result['more_meanings'] = more_meanings
        # print(f'more_meanings: {more_meanings}')
        # # ------------------------------

        # result['meaning']
        # result['word'] = word
        # result['raw_example'] = example
        # result['raw_meaning'] = ref_meaning

        transformed_data.append(result)

    save_dataset(transformed_data, output_filename)




def sort_and_save(filename, output_filename):
    data_list = read_jsonl(filename)
    sorted_data = sort_and_filter_data(data_list, by_key='upvotes')
    save_dataset(sorted_data, output_filename)


if __name__ == "__main__":
    # filename = './data/words.json'
    # clean_data(filename, 10)
    filename = './data/words_raw.json'
    sorted_filename = './data/words_sorted.json'
    cleaned_filename = './data/words_cleaned.json' # remove nsfw contents
    knowledge_checked_filename = './data/words_knowledge_checked.json' # check knowledge
    transformed_filename = './data/words_transformed_all_fewshot_v2_all.json' # transform to strict dictionary style
    counterfactual_transformed_filename = './data/words_counterfactual_transformed_all_fewshot_mini10.json' # transform to strict dictionary style

    # sort_and_save(filename, sorted_filename)
    # clean_data(sorted_filename, cleaned_filename, sample_num=1000)
    # print(f'Total cost: ${LargeLanguageModel.calculate_total_cost():.6f}')
    # check_knowledge_and_save(cleaned_filename, knowledge_checked_filename)
    # print(f'Total cost: ${LargeLanguageModel.calculate_total_cost():.6f}')
    # transform_data(knowledge_checked_filename, transformed_filename)
    counterfactual_transform(transformed_filename, counterfactual_transformed_filename)
    print(f'Total cost: ${LargeLanguageModel.calculate_total_cost():.6f}')
