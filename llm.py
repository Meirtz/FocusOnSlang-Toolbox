import os
import openai
# print(openai.__version__)
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configurations and constants
MODEL_CONFIG = {
    "GPT_LIST": ["ada", "babbage", "curie", "davinci", "text-ada-001", "text-babbage-001", "text-curie-001",
                 "text-davinci-001", "text-davinci-002", "text-davinci-003", "gpt-3.5-turbo-instruct"],
    "CHAT_GPT_LIST": ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"],
    "LLAMA_LIST": ["llama-7b", "llama-13b", "llama-30b", "llama-65b", "alpaca-7b", "alpaca-13b", "vicuna-7b", "vicuna-13b"],
    "OTHER_7B_LIST": ["zephyr-7b", "mistral-7b", "llama-7b-chat"],
    "OTHER_7B_HUGGINGFACE_MODEL_PATH": {
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "llama-7b-chat": "meta-llama/Llama-2-7b-chat-hf"
    },
    "COSTS": {
        'gpt-4-1106-preview': {'input': 0.001 / 1000, 'output': 0.03 / 1000},
        'gpt-3.5-turbo-instruct': {'input': 0.0015 / 1000, 'output': 0.002 / 1000},
        'gpt-3.5-turbo-1106': {'input': 0.0010 / 1000, 'output': 0.0020 / 1000},
        'gpt-3.5-turbo': {'input': 0.0015 / 1000, 'output': 0.002 / 1000},
        'gpt-4': {'input': 0.03 / 1000, 'output': 0.06 / 1000}
    }
}


class LargeLanguageModel:
    """ A class for handling interactions with large language models like OpenAI's GPT and LLaMA. """

    model_token_usage = {}

    def __init__(self, model_name='text-davinci-003', max_length=512, temperature=0.7, use_openai=True, openai_api_key=None, llama_model_path=None):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.use_openai = use_openai
        self.openai_api_key = self._get_openai_api_key(openai_api_key)
        self.client = OpenAI(api_key=self.openai_api_key) if self.use_openai else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.tokenizer, self.pipeline = self._initialize_model(llama_model_path)

    @staticmethod
    def _get_openai_api_key(provided_key):
        """ Retrieves the OpenAI API key from environment variable or parameter. """
        env_api_key = os.getenv('OPENAI_API_KEY')
        if provided_key is not None:
            return provided_key
        elif env_api_key is not None:
            return env_api_key
        else:
            raise ValueError("OpenAI API key is required when use_openai is True")

    def _initialize_model(self, llama_model_path):
        """ Initializes the model based on the model name. """
        if self.use_openai:
            openai.api_key = self.openai_api_key
            return None, None, None
        else:
            if self.model_name in MODEL_CONFIG['LLAMA_LIST']:
                model_path = llama_model_path or f"meta-llama/Llama-2-{self.model_name.split('-')[1]}-hf"
                tokenizer = LlamaTokenizer.from_pretrained(model_path)
                model = LlamaForCausalLM.from_pretrained(model_path, use_flash_attention_2=True, torch_dtype=torch.bfloat16, device_map=self.device)
                tokenizer.pad_token = tokenizer.eos_token
                pipeline = self._create_pipeline(model, tokenizer)
                return model, tokenizer, pipeline
            elif self.model_name in MODEL_CONFIG['OTHER_7B_LIST']:
                model_path = llama_model_path or MODEL_CONFIG['OTHER_7B_HUGGINGFACE_MODEL_PATH'][self.model_name]
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path, use_flash_attention_2=True, torch_dtype=torch.bfloat16, device_map=self.device)
                tokenizer.pad_token = tokenizer.eos_token
                pipeline = self._create_pipeline(model, tokenizer)
                return model, tokenizer, pipeline

    def _create_pipeline(self, model, tokenizer):
        """ Creates a text generation pipeline. """
        return pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map='auto')

    # @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=5), retry_error_callback=lambda retry_state: LargeLanguageModel.log_retry_error(retry_state))
    def get_response(self, prompt, mode='text'):
        """ Retrieves a response from the LLM based on the provided prompt. """
        return self.get_openai_response(prompt, mode) if self.use_openai else self.get_llama_response(prompt)

    def get_openai_response(self, prompt, mode):
        """ Retrieves a response from OpenAI's API. """
        if self.model_name in MODEL_CONFIG['GPT_LIST']:
            return self.get_text_completion_response(prompt, mode)
        elif self.model_name in MODEL_CONFIG['CHAT_GPT_LIST']:
            return self.get_chat_completion_response(prompt, mode)

    def get_text_completion_response(self, prompt):
        """ Retrieves a text completion response from OpenAI's API. """
        response = self.client.completions.create(model=self.model_name, prompt=prompt, max_tokens=self.max_length, temperature=self.temperature)
        self._update_token_counts(response['usage']['prompt_tokens'], response['usage']['total_tokens'])
        return response.choices[0].text.strip()

    def get_chat_completion_response(self, prompt, mode):
        """ Retrieves a chat completion response from OpenAI's API. """
        response = self.client.chat.completions.create(model=self.model_name,
                                                       response_format={'type': mode},
                                                       messages=[{"role": "user", "content": prompt}])
        self._update_token_counts(response.usage.prompt_tokens, response.usage.total_tokens)
        return response.choices[0].message.content.strip()

    def get_llama_response(self, prompt):
        """ Retrieves a response from a LLaMA model. """
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=self.max_length, temperature=self.temperature, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id, do_sample=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(prompt)

    def _update_token_counts(self, prompt_tokens, total_tokens):
        """ Updates the token usage counts for the model. """
        if self.model_name not in self.model_token_usage:
            self.model_token_usage[self.model_name] = {'prompt': 0.0, 'completion': 0.0}
        self.model_token_usage[self.model_name]['prompt'] += prompt_tokens
        self.model_token_usage[self.model_name]['completion'] += (total_tokens - prompt_tokens)

    @staticmethod
    def log_retry_error(retry_state):
        """
        Logs the error that caused a retry.

        :param retry_state: The state object passed by tenacity.
        """
        logger.error(f"Attempt {retry_state.attempt_number}: An error occurred: {retry_state.outcome.exception()}")
        # Optionally, log more details or take additional actions here
        print(f"Attempt {retry_state.attempt_number}: An error occurred: {retry_state.outcome.exception()}")

    @classmethod
    def calculate_total_cost(cls):
        """ Calculates the total cost of token usage for all models. """
        total_cost = 0
        for model_name, token_counts in cls.model_token_usage.items():
            model_cost = MODEL_CONFIG['COSTS'].get(model_name)
            if model_cost:
                total_cost += (model_cost['input'] * token_counts['prompt']) + (model_cost['output'] * token_counts['completion'])
        return total_cost


if __name__ == '__main__':
    llm = LargeLanguageModel(use_openai=True, model_name="gpt-3.5-turbo-1106")
    response = llm.get_response('Translate the following English text to French: "Hello, how are you?"')
    print(response)
    print(f'Total cost: ${LargeLanguageModel.calculate_total_cost():.6f}')
