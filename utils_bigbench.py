import json
import pandas as pd
import numpy as np
import re


def get_reasoning_data_bigbench(start=0):
    with open('./data/bigbench_task.json') as f:
        data = json.load(f)
        return data['examples'][start:]


def generate_prompt_bigbench(data):
    prompt = f"""
        Question: {data['input']}
        Options: 
    """
    for option in range(len(data['target_scores'])):
        prompt += f"""{option}) {list(data['target_scores'].keys())[option]} \n"""

    prompt += "There will be always one correct option and one correct option only. At the end of the response, please use curly brackets to mark your answer, like {3}. "

    prompt_a_last = prompt + "Please give out the reasoning logic first and then answer the question by selecting the options."
    prompt_a_first = prompt + "Please give out the correct option in the first sentence and then give out the logic."
    prompt += "Please give out the correct option. "

    return prompt, prompt_a_first, prompt_a_last


def generate_review_prompt_bigbench(data, result1, result2):
    prompt, _, _ = generate_prompt_bigbench(data)
    prompt += "\n Each time I asked you twice, once I asked you to give me the answer first then the logic, once I asked you to give me the logic first then the answer, and sometimes the two answers are different. Here I want you to review the logic of the two results and give me the final answer. "
    prompt += "Result 1: \n"
    prompt += result1 + "\n"
    prompt += "Result 2: \n"
    prompt += result2 + "\n  Still, there will be always one correct option and one correct option only. Please use curly brackets to mark your answer, like {3}."
    return prompt


def experiment_bigbench(get_result_function, start=0):
    # parameter "start" for resume experiment if disconnected
    json_objects = get_reasoning_data_bigbench(start)
    result_df = pd.DataFrame([], columns=['result', 'result_a_first', 'result_a_last', 'result_after_review',
                                          'correct_answer', 'result_choice', 'result_a_first_choice',
                                          'result_a_last_choice'])
    name = get_result_function.__name__.split('_')[1]
    for i, data in enumerate(json_objects):
        prompt, prompt_a_first, prompt_a_last = generate_prompt_mmlu(data)

        result = get_result_function(prompt)
        if result is None:
            continue
        result_choice = analyse_results(result)
        result_a_first = get_result_function(prompt_a_first)
        if result_a_first is None:
            continue
        result_a_first_choice = analyse_results(result_a_first)
        result_a_last = get_result_function(prompt_a_last)
        if result_a_last is None:
            continue
        result_a_last_choice = analyse_results(result_a_last)
        review_prompt = generate_review_prompt_truthfulqa(data, result_a_last, result_a_first)
        review_result = get_result_function(review_prompt)
        review_choice = analyse_results(review_result)
        if review_choice is None:
            continue

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
            result_df.to_csv('result_{}_mmlu.csv'.format(name.split('_')[-1]))
        else:
            result_df.to_csv('result_{}_{}_mmlu.csv'.format(name.split('_')[-1], start))


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
