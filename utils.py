import json
import jsonlines
import re

def replace_ignore_case(string, old, new):
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, string)

def read_jsonl(filename):
    data_list = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def sort_and_filter_data(data_list, by_key):
    # Filter out entries with upvotes less than 2
    filtered_data = [x for x in data_list if x.get(by_key, 0) >= 2]

    # Sort the remaining data based on the value of the key "upvotes" in descending order
    sorted_data = sorted(filtered_data, key=lambda x: x.get(by_key, 0), reverse=True)

    return sorted_data


def save_dataset(data, output_filename):
    # Create a new dictionary with 'dataset' as the key
    new_data = {'dataset': data}

    # Write the sorted data to a new json file
    with open(output_filename, 'w') as file:
        json.dump(new_data, file, indent=4)

def read_json(filename):
    with open(filename, "r", encoding="utf-8") as fin:
        data_dict = json.load(fin)
    data_list = data_dict["dataset"]
    return data_list


if __name__ == "__main__":
    filename_selfaware = "./data/SelfAware.json"
    data_list = read_json(filename_selfaware)
    # Description of the data
    print("The data is a list of dict, each dict has the following keys:")
    print(data_list[0].keys())
    print("The data has {} examples".format(len(data_list)))
    print("The first example is:")
    print(data_list[0])
    print("The second example is:")
    print(data_list[1])
    print("The third example is:")
    print(data_list[2])

# The data is a list of dict, each dict has the following keys:
# dict_keys(['question_id', 'question', 'answer', 'answerable', 'source'])
# The data has 3369 examples
# The first example is:
# {'question_id': 1, 'question': "What form of entertainment are 'Slow Poke' and 'You Belong to Me'?", 'answer': [
# 'song'], 'answerable': True, 'source': 'hotpot_train'}
# The second example is:
# {'question_id': 2, 'question': "A person's identity is defined as the totality of what?", 'answer': ["one's
# self-construal"], 'answerable': True, 'source': 'squadqa_train'}
# The third example is:
# {'question_id': 3, 'question': 'Which breed of dog is bigger, Japanese Terrier or Cane Corso?', 'answer': ['cane
# corso'], 'answerable': True, 'source': 'hotpot_train'}


    filename_words = "./data/words.json"
    data_list = read_json(filename_words)
    # Description of the data
    print("The data is a list of dict, each dict has the following keys:")
    print(data_list[0].keys())
    print("The data has {} examples".format(len(data_list)))
    print("The first example is:")
    print(data_list[0])
    print("The second example is:")
    print(data_list[1])
    print("The third example is:")
    print(data_list[2])
