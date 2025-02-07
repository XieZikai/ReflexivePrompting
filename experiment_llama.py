from llamaapi import LlamaAPI
from utils_logiqa import *
from utils_truthfulqa import *
from utils_mmlu import *
from utils_bigbench import *
from openai import OpenAI


API_KEY = '<PLEASE_USE_YOUR_API_KEY>'
llama = LlamaAPI(API_KEY)
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.llama-api.com"
)


def get_llama_result(prompt):
    api_request_json = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "model": "llama3.1-70b",
        "max_tokens": 500,
        "stream": False,
    }
    response = llama.run(api_request_json)
    return response.json()['choices'][0]['message']['content']


if __name__ == "__main__":
    experiment_truthfulqa(get_llama_result)
    experiment_mmlu(get_llama_result)
    experiment_logiqa(get_llama_result)
    experiment_bigbench(get_claude_result)
