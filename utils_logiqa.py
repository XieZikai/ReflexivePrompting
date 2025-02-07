import json
import re
import pandas as pd


# To download LogiQA dataset, please visit: https://github.com/csitfun/LogiQA2.0
def get_reasoning_data_logiqa(start=0, clip=1000):
    file = './data/LogiQA2.0-main/logiqa/DATA/LOGIQA/train.txt'
    json_objects = []

    count = start
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    json_data = json.loads(line)
                    json_objects.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in line: {line}\n{e}")
            count += 1
            if clip is not None and count >= clip:
                break

    return json_objects


def generate_prompt_logiqa(data):
    prompt = f"""
        Context: {data['text']}
        Question: {data['question']}
        Options: """
    for option in range(len(data['options'])):
        prompt += f"""{option}) {data['options'][option]} \n"""

    prompt += "There will be always one correct option and one correct option only. At the end of the response, please use parantheses to mark your answer, like {3}. "

    prompt_a_last = prompt + "Please give out the reasoning logic first and then answer the question by selecting the options."
    prompt_a_first = prompt + "Please give out the correct option in the first sentence and then give out the logic."
    prompt += "Please give out the correct option. "

    return prompt, prompt_a_first, prompt_a_last


def generate_review_prompt_logiqa(data, result1, result2):
    prompt, _, _ = generate_prompt_logiqa(data)
    prompt += "\n Each time I asked you twice, once I asked you to give me the answer first then the logic, once I asked you to give me the logic first then the answer, and sometimes the two answers are different. Here I want you to review the logic of the two results and give me the final answer. "
    prompt += "Result 1: \n"
    prompt += result1 + "\n"
    prompt += "Result 2: \n"
    prompt += result2 + "\n  Still, there will be always one correct option and one correct option only. Please use parantheses to mark your answer, like {3}."
    return prompt


def experiment_logiqa(get_result_function, start=0, clip=1000):
    # parameter "start" for resume experiment if disconnected
    json_objects = get_reasoning_data_logiqa(start, clip)
    result_df = pd.DataFrame([], columns=['result', 'result_a_first', 'result_a_last', 'result_after_review',
                                          'correct_answer', 'result_choice', 'result_a_first_choice',
                                          'result_a_last_choice'])
    name = get_result_function.__name__.split('_')[1]

    for i, data in enumerate(json_objects):
        prompt, prompt_a_first, prompt_a_last = generate_prompt_logiqa(data)

        result = get_result_function(prompt)
        result_choice = analyse_results(result)
        result_a_first = get_result_function(prompt_a_first)
        result_a_first_choice = analyse_results(result_a_first)
        result_a_last = get_result_function(prompt_a_last)
        result_a_last_choice = analyse_results(result_a_last)
        review_prompt = generate_review_prompt_logiqa(data, result_a_last, result_a_first)
        review_result = get_result_function(review_prompt)
        review_choice = analyse_results(review_result)

        result_df = result_df._append({
            'result': result,
            'result_a_first': result_a_first,
            'result_a_last': result_a_last,
            'result_review': review_result,
            'correct_answer': data['answer'],
            'result_choice': result_choice,
            'result_a_first_choice': result_a_first_choice,
            'result_a_last_choice': result_a_last_choice,
            'result_review_choice': review_choice
        }, ignore_index=True)

        if start == 0:
            result_df.to_csv('result_{}_logiqa.csv'.format(name.split('_')[-1]))
        else:
            result_df.to_csv('result_{}_{}_logiqa.csv'.format(name.split('_')[-1], start))


def analyse_results(result):
    pattern = r'\{\d\}'
    pattern_backup_1 = r'\{\d'
    pattern_backup_2 = r'\d\}'
    re_result = re.findall(pattern, result)
    if len(re_result) > 0:
        return re_result[0][1]
    else:
        re_result_1 = re.findall(pattern_backup_1, result)
        re_result_2 = re.findall(pattern_backup_2, result)
        if len(re_result_1) > 0:
            return re_result_1[0][1]
        if len(re_result_2) > 0:
            return re_result_2[0][0]
        print('No result matching found, could be error. ')
        print('Result: ', result)
        return '9'
