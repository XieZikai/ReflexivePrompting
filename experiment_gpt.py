from openai import OpenAI
from utils_logiqa import *
from utils_truthfulqa import *
from utils_mmlu import *
import os


API_KEY = '<PLEASE_USE_YOUR_API_KEY>'
os.environ['OPENAI_API_KEY'] = API_KEY
client = OpenAI()


def get_gpt_result(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    experiment_logiqa(get_gpt_result)
    experiment_truthfulqa(get_gpt_result)
    experiment_mmlu(get_gpt_result)
