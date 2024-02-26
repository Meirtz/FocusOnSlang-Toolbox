from prompt import *
from utils import *
from agents import *
import random
from tqdm import tqdm
import datetime
import time
import os
from eval_urban_f1 import calculate_metrics
from eval_urban_simcse import calculate_similarity_and_judge
import re

# Urban Dictionary Templates
URBAN_DICTIONARY_TEMPLATES = {
    'Direct': lambda prompt_entity: urban_direct_prompt.format(*prompt_entity),
    'Instruction': lambda prompt_entity: urban_instruction_prompt.format(word=prompt_entity[0], example=prompt_entity[1]),
    'ICL': lambda prompt_entity: urban_icl_prompt.format(word=prompt_entity[0], example=prompt_entity[1]),
    'CoT': lambda prompt_entity: urban_cot_prompt.format(word=prompt_entity[0], example=prompt_entity[1]), # plain guess
    'Causal': lambda prompt_entity: urban_causal_prompt.format(prompt_entity[1].replace(prompt_entity[0], '[MASKRD_PHRASE]')), # masked guess
    'GenerateNewSentences': lambda prompt_entity: urban_causal_mid_prompt.format(word=prompt_entity[0], example=prompt_entity[1]), # generate new sentences
    'SummarizeResult': lambda prompt_entity: urban_causal_final_prompt.format(*prompt_entity),
    # Add other policies if needed
    'Causal_Propose': lambda prompt_entity: urban_causal_legacy_propose_prompt.format(word=prompt_entity[0], example=prompt_entity[1]),
    'Causal_CoT': lambda prompt_entity: urban_causal_legacy_cot_prompt.format(phrase=prompt_entity[0],
                                                                              example=prompt_entity[1],
                                                                              reconstructed_example=prompt_entity[2],
                                                                              entity_candidates_json=prompt_entity[3])
}

class CommonAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()

    def execute(self, agent, prompt):
        current_joint_memory = {}
        raw_result = agent.get_response(prompt)
        current_joint_memory['plain_guess'] = agent.short_memory
        return raw_result, current_joint_memory


class CausalAlgorithm(Algorithm):
    def __init__(self, llm):
        super().__init__()
        self.causal_propose_agent = Agent(llm, parser=causal_propose_parser)
        self.causal_cot_agent = Agent(llm, parser=cot_parser)
        self.causal_propose_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal_Propose'])
        self.causal_cot_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal_CoT'])

    def execute(self, word_with_example):
        current_joint_memory = {}
        
        causal_propose_prompt = self.causal_propose_task.get_prompt(word_with_example)
        causal_propose_response = self.causal_propose_agent.get_response(causal_propose_prompt)
        current_joint_memory['causal_propose'] = self.causal_propose_agent.short_memory
        entity_list = causal_propose_response['entity_replacement_list']
        recon_example = causal_propose_response['reconstructed_example']
        print("Causal Propose Response: {}".format(causal_propose_response))
        # time.sleep(1)

        causal_cot_prompt = self.causal_cot_task.get_prompt((word_with_example[0], word_with_example[1], entity_list, recon_example))
        causal_cot_response = self.causal_cot_agent.get_response(causal_cot_prompt)
        current_joint_memory['causal_cot'] = self.causal_cot_agent.short_memory
        print("Causal CoT Response: {}".format(causal_cot_response))
        # time.sleep(1)

        return causal_cot_response.replace('[MASKED_PHRASE]', word_with_example[0]) , current_joint_memory

        
class FOCUSAlgorithm(Algorithm):
    def __init__(self, llm):
        super().__init__()
        self.guess_agent = Agent(llm, parser=cot_parser)
        self.masked_guess_agent = Agent(llm, parser=masked_guess_parser)
        self.generate_agent = Agent(llm, parser=generate_new_sentences_parser)
        self.summarize_agent = Agent(llm, parser=summarize_parser)
        self.plain_guess_task = Task(URBAN_DICTIONARY_TEMPLATES['CoT'])
        self.masked_guess_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal'])
        self.generate_task = Task(URBAN_DICTIONARY_TEMPLATES['GenerateNewSentences'])
        self.generate_guess_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal'])
        self.summarize_task = Task(URBAN_DICTIONARY_TEMPLATES['SummarizeResult'])
        

    def execute(self, word_with_example):

        current_joint_memory = {}
        # Plain guess
        plain_guess_prompt = self.plain_guess_task.get_prompt(word_with_example)
        plain_guess_response = self.guess_agent.get_response(plain_guess_prompt)
        current_joint_memory['plain_guess'] = self.guess_agent.short_memory
        print("Plain Guess Response: {}".format(plain_guess_response))
        # time.sleep(1)

        # # Masked guess
        MASKRD_PHRASE_with_example = (word_with_example[0], replace_ignore_case(word_with_example[1], word_with_example[0], '[MASKRD_PHRASE]'))
        masked_guess_prompt = self.masked_guess_task.get_prompt(MASKRD_PHRASE_with_example)
        masked_guess_response = self.masked_guess_agent.get_response(masked_guess_prompt)
        current_joint_memory['masked_guess'] = self.masked_guess_agent.short_memory
        print("Masked Guess Response: {}".format(masked_guess_response))
        # time.sleep(1)

        # Generate new sentences
        generate_prompt = self.generate_task.get_prompt(word_with_example)
        generate_response = self.generate_agent.get_response(generate_prompt)
        current_joint_memory['generate'] = self.generate_agent.short_memory
        print("Generate Response: {}".format(generate_response))

        # entity_replaced_prompt = self.generate_guess_task.get_prompt((word_with_example[0], generate_response))
        # entity_replaced_response = self.masked_guess_agent.get_response(entity_replaced_prompt)
        # # current_joint_memory['generate_guess'] = self.masked_guess_agent.short_memory
        # print("Generate Guess Response: {}".format(entity_replaced_response))

        # time.sleep(1)

        # Summarize 
        sentences = (word_with_example[0], word_with_example[1], plain_guess_response, masked_guess_response, generate_response)
        summarize_prompt = self.summarize_task.get_prompt(sentences)
        summarize_response = self.summarize_agent.get_response(summarize_prompt)
        current_joint_memory['summarize'] = self.summarize_agent.short_memory
        print("Summarize Response: {}".format(summarize_response))
        # time.sleep(1)
        
        return summarize_response, current_joint_memory


# Defining specific parsing functions
def cot_parse_function(response):
    return response.split("Conclusion:")[1].strip()

def icl_parse_function(response):
    return response.split("Meaning:")[-1].strip()

def generate_new_sentences_parse_function(response):
    return response.split("Reconstructed Sentence:")[-1].strip()

def masked_guess_parse_function(response):
    return response.split("Inferred Meaning:")[-1].strip()

def causal_propose_parse_function(response):
    # Extracts the content for both entity replacement list and reconstructed example
    results = {}

    # Extract entity replacement list
    match_entity_list = re.search(r'\{([^}]*)\}', response, re.DOTALL)
    if match_entity_list:
        results['entity_replacement_list'] = match_entity_list.group(1).strip()
    else:
        results['entity_replacement_list'] = "No entity replacement list found."

    # Extract reconstructed example
    match_reconstructed_example = re.search(r'Reconstructed Sentence:\s*(.*?)\s*(?:\n|$)', response, re.DOTALL)
    if match_reconstructed_example:
        results['reconstructed_example'] = match_reconstructed_example.group(1).strip()
    else:
        results['reconstructed_example'] = "No reconstructed example found."

    return results
# Example usage:
# response = "Your response text with {Entity Replacement List content}"
# parsed_content = parse_entity_replacement_list(response)


def summarize_parse_function(response):
    if "\nMeaning: " in response:
        return response.split("\nMeaning: ")[-1].strip()
    elif "\nDefinition: " in response:
        return response.split("\nDefinition: ")[-1].strip()
    else:
        return response
    

# def default_parser(message):
#     print("Calling default_parser")
#     result = message.split("\n")[0]
#     if len(result) == 0:
#         result = message.split("\n")[1]
#         print("Empty result, use second line: {}".format(result))
#     return result

def default_parser(message):
    return message

# Creating instances of Parser with specific parsing functions
cot_parser = Parser(cot_parse_function)
icl_parser = Parser(icl_parse_function)
masked_guess_parser = Parser(masked_guess_parse_function)
generate_new_sentences_parser = Parser(generate_new_sentences_parse_function)
summarize_parser = Parser(summarize_parse_function)
causal_propose_parser = Parser(causal_propose_parse_function)

def run(filename, model_name='gpt-3.5-turbo', sample_num=3, template_name='Direct'):

    use_focus = template_name == 'FOCUS'
    use_causal = template_name == 'Causal'

    data_list = read_json(filename)
    if sample_num == 'all':
        sample_index = range(len(data_list))
    elif type(sample_num) is str and ':' in sample_num:
        sample_index = range(int(sample_num.split(':')[0]), int(sample_num.split(':')[1]))
        sample_num = sample_num.replace(':', '-')
    else:
        sample_index = range(sample_num)

    all_results = []

    # File naming based on model name and sample number
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"outputs/Urban_{model_name}_{template_name}_{timestamp}_{sample_num}.json"
    print(f"Output file: {output_filename}")

    parser = Parser(default_parser)
    if template_name == 'ICL':
        parser = icl_parser
        print("Using ICL parser")
    elif template_name == 'CoT':
        parser = cot_parser
        print("Using CoT parser")

    # read file
    data_list = read_json(filename)

    # Initialize LargeLanguageModel
    llm = LargeLanguageModel(use_openai=True, model_name=model_name)

    templates = URBAN_DICTIONARY_TEMPLATES
    # algorithm = CommonAlgorithm() if not use_focus else FOCUSAlgorithm(llm)
    if template_name == 'Causal':
        algorithm = CausalAlgorithm(llm)
    elif template_name == 'FOCUS':
        algorithm = FOCUSAlgorithm(llm)
    else:
        algorithm = CommonAlgorithm()


    all_results = []
    mean_accumulate_scores = {
        'exact_match': 0.0,
        'f1_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'bleu_score': 0.0,
        'simcse_score': 0.0
    }

    for i in tqdm(sample_index):
        word_with_example = (data_list[i]['word'], data_list[i]['example'])#, data_list[i]['raw_example'])
        ref_meaning = data_list[i]['meaning']

        enable_bleu_retry = False
        best_prediction = None
        best_score = 0
        retry_count = 0
        max_retries = 5

    
        if use_focus or use_causal:
            prediction, current_joint_memory  = algorithm.execute(word_with_example)
            
        else:
            template_function = templates[template_name] 
            # Create a Task instance with only the template function
            task = Task(template_function)

            # Generate the prompt using word_with_example
            prompt = task.get_prompt(word_with_example)

            # Create an Agent instance with the LargeLanguageModel instance and the parser
            agent = Agent(llm, parser=parser)

            # Get the response using the algorithm
            prediction, current_joint_memory = algorithm.execute(agent, prompt)
            # time.sleep(1)
    

        result = {
            'word': data_list[i]['word'],
            # 'prompt': task.get_prompt(word_with_example),
            'meaning': ref_meaning,
            'prediction': prediction,
            'memory': current_joint_memory,
            'scores': calculate_metrics(prediction, ref_meaning)
        }
        # current_scores = calculate_metrics(prediction, ref_meaning)
        # for k in mean_accumulate_scores.keys():
        #     mean_accumulate_scores[k] += current_scores[k]
        # print("Current scores: {}".format(current_scores))
        # print("Mean scores: {}".format({k: v/(i+1) for k, v in mean_accumulate_scores.items()}))

        all_results.append(result)

        print(f'Current cost: ${LargeLanguageModel.calculate_total_cost():.6f}')

        # Write to the same output file
        with open(output_filename, "w") as outfile:
            json.dump(all_results, outfile, indent=4)

    # Write the aggregated results to a JSON file
    new_data = {'output': all_results}
    with open("outputs/aggregated_output.json", "w") as outfile:
        json.dump(new_data, outfile, indent=4)

# Usage
if __name__ == "__main__":
    run("data\words_transformed_all_fewshot_v2_all.json",
        model_name='gpt-3.5-turbo-1106',
        sample_num="300:357",
        template_name='Direct')
    print(f'Total cost: ${LargeLanguageModel.calculate_total_cost():.6f}')
  