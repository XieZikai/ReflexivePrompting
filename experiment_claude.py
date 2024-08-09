import boto3
import os
import anthropic
from utils_logiqa import *
from utils_truthfulqa import *
from utils_mmlu import *


API_KEY = '<PLEASE_USE_YOUR_API_KEY>'
os.environ['ANTHROPIC_API_KEY'] = API_KEY
brt = boto3.client(service_name='bedrock-runtime', region_name='eu-west-2')
client = anthropic.Anthropic()


def get_claude_result(prompt):
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful assistant.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text


if __name__ == "__main__":
    experiment_logiqa(get_claude_result)
    experiment_truthfulqa(get_claude_result)
    experiment_mmlu(get_claude_result)
