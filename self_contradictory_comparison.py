import json
from openai import OpenAI
import os


API_KEY = '<PLEASE_USE_YOUR_API_KEY>'
os.environ['OPENAI_API_KEY'] = API_KEY
client = OpenAI()


def get_gpt_result(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Convert the JSON-formatted text to a Python dictionary
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def generate_prompt(txt_file):
    a_first = "I will give you some context and I want you to complete a sentence for me. Please give the answer then give me the logic. \n\n"
    l_first = "I will give you some context and I want you to complete a sentence for me. Please give the logic then give me the answer. \n\n"

    a_first += "Context: " + txt_file['prefix'] + "\n"
    l_first += "Context: " + txt_file['prefix'] + "\n"

    a_first += "Sentence to complete: " + txt_file['triple'][0][0] + " " + txt_file['triple'][0][1] + "\n"
    l_first += "Sentence to complete: " + txt_file['triple'][0][0] + " " + txt_file['triple'][0][1] + "\n"

    return a_first, l_first


def experiment(directory='./sentences/chatgpt'):
    dirs = os.listdir(directory)
    for folder in dirs:
        for name in ['0_0_0.txt', '0_1_0.txt', '0_2_0.txt', '0_3_0.txt']:
            if os.path.exists(os.path.join(directory, folder, 'm3', name)):
                break
        file = os.path.join(directory, folder, 'm3', name)
        a_first, l_first = generate_prompt(read_txt_file(file))
        a_result = get_gpt_result(a_first)
        l_result = get_gpt_result(l_first)
        result = {
            'a_first': a_first,
            'l_first': l_first,
            'a_result': a_result,
            'l_result': l_result,
        }
        with open('./data/{}.json'.format(folder), 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    experiment()
