from llm import LargeLanguageModel, MODEL_CONFIG
from utils import *
from prompt import *
from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self, template_function):
        """
        Initializes a Task object with a template function.

        Args:
            template_function (function): A function to generate a prompt based on a given raw prompt.
        """
        self.template_function = template_function

    def get_prompt(self, raw_prompt):
        """
        Generates a prompt based on the template function and the provided raw prompt.

        Args:
            raw_prompt (str): The raw prompt to be used for generating the final prompt.

        Returns:
            str: The generated prompt based on the template function and the raw prompt.
        """
        return self.template_function(raw_prompt)


class Parser:
    """
    A class for response parsers that takes a parsing function during initialization.
    """

    def __init__(self, parse_function):
        """
        Initializes the Parser with a given parsing function.

        :param parse_function: A function that takes a response string and returns a parsed result.
        """
        self.parse_function = parse_function

    def parse(self, response):
        """
        Uses the provided parsing function to parse the response.

        :param response: The response string to be parsed.
        :return: The result of the parsing function.
        """
        try:
            return self.parse_function(response)
        except Exception as e:
            print(f"Failed to parse response: {e}. Returning unmodified response.")
            return response


class Algorithm(ABC):
    def __init__(self, agent=None):
        self.agent = agent

    @abstractmethod
    def execute(self, task):
        pass


class Agent:
    def __init__(self, llm, parser=None):
        self.llm = llm
        self.parser = parser if parser is not None else Parser(lambda x: x)
        self.short_memory = ""
        if parser is None:
            print("Warning: No parser provided for agent. Using default parser.")

    def get_response(self, prompt):
        # prompt = task.get_prompt()
        self.short_memory = self.llm.get_response(prompt)
        return self.parser.parse(self.short_memory)


if __name__ == "__main__":

    def score_extraction_function(response):
        try:
            print(f'Parsing response: {response}\n')
            # Find the score in the format "Score: [number]" in the response
            score_str = response.split('Score: ')[-1].split()[0]  # Gets the part after "Score: " and before any space
            score = float(score_str)
            print(f'Extracted score: {score}\n')
            return score
        except (IndexError, ValueError):
            # Handle cases where parsing fails
            return 0


    score_parser = Parser(score_extraction_function)


    class TreeOfThoughtsAlgorithm(Algorithm):
        def __init__(self, templates, num_candidates=5, depth=5):
            super().__init__()
            self.templates = templates
            self.num_candidates = num_candidates
            self.depth = depth

        def execute(self, task, raw_prompt):
            queue = [(raw_prompt, 0, None)]
            best_thoughts = []

            while queue:
                current_prompt, current_depth, parent_prompt = queue.pop(0)
                if current_depth >= self.depth:
                    best_thoughts.append(current_prompt)
                    continue

                candidates = self.generate_candidates(task, current_prompt)
                scored_candidates = [(cand, self.evaluate_candidate(task, current_prompt, cand)) for cand in candidates]
                best_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:2]

                for candidate, _ in best_candidates:
                    queue.append((candidate, current_depth + 1, current_prompt))

            return best_thoughts

        def generate_candidates(self, task, raw_prompt):
            modified_prompt = task.get_prompt(raw_prompt)
            response = self.agent.llm.get_response(modified_prompt)
            return response.split('\n')[:self.num_candidates] if response else []

        def evaluate_candidate(self, task, raw_prompt, candidate):
            evaluation_prompt = task.get_prompt(raw_prompt)
            response = self.agent.llm.get_response(evaluation_prompt)
            return score_parser.parse(response) if response else 0


    CREATIVE_WRITING_TEMPLATES = {
        'EvaluateThought': lambda thought, words:
        f"Evaluate the creativity and coherence of this text. "
        f"Check if it contains the words '{words[0]}' and '{words[1]}'. "
        f"Append a score between 1 and 10 at the end in the format 'Score: [number]': {thought}",
        'GenerateThoughts': lambda
            words: f"Write a short text that starts with '{words[0]}' and ends with '{words[1]}'. Provide five different ideas.",

    }

    llm = LargeLanguageModel(use_openai=True, model_name="gpt-3.5-turbo-1106")
    agent = Agent(llm)

    tot_algo = TreeOfThoughtsAlgorithm(CREATIVE_WRITING_TEMPLATES)
    tot_algo.agent = agent  # Set the agent for the algorithm

    raw_prompt = "Using words 'sunrise' and 'sunset', describe a scene."
    creative_task = Task(lambda words: CREATIVE_WRITING_TEMPLATES['GenerateThoughts'](words))

    creative_texts = tot_algo.execute(creative_task, raw_prompt)

    for text in creative_texts:
        print(text)

    print(f'Total cost: ${LargeLanguageModel.calculate_total_cost():.6f}')

