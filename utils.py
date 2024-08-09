import re


def result_df_correction(df):
    for i in range(len(df)):
        if str(df.iloc[i]['result_choice']) == '9':
            df.at[i, 'result_choice'] = result_correction(df.iloc[i]['result'])
        if str(df.iloc[i]['result_a_first_choice']) == '9':
            df.at[i, 'result_a_first_choice'] = result_correction(df.iloc[i]['result_a_first'])
        if str(df.iloc[i]['result_a_last_choice']) == '9':
            df.at[i, 'result_a_last_choice'] = result_correction(df.iloc[i]['result_a_last'])
        if str(df.iloc[i]['result_review_choice']) == '9':
            df.at[i, 'result_review_choice'] = result_correction(df.iloc[i]['result_review'])


def result_correction(result):
    """
    Detect different bracket symbol and 'None of above' results for LLM outputs
    """
    first_sentence = result.split('\n')[0]
    last_sentence = result.split('\n')[-2]
    pattern1 = r'\(\d\)'
    pattern2 = r'\d\)'
    last_sentence_matching_1 = re.findall(pattern1, last_sentence)
    last_sentence_matching_2 = re.findall(pattern2, last_sentence)
    first_sentence_matching_1 = re.findall(pattern1, first_sentence)
    first_sentence_matching_2 = re.findall(pattern2, first_sentence)

    if len(last_sentence_matching_1) > 0:
        return last_sentence_matching_1[0][1]
    elif len(last_sentence_matching_2) > 0:
        return last_sentence_matching_2[0][0]
    elif len(first_sentence_matching_1) > 0:
        return first_sentence_matching_1[0][1]
    elif len(first_sentence_matching_2) > 0:
        return first_sentence_matching_2[0][0]
    return 9


def analysis(df):
    df['correctness_raw'] = df.apply(lambda row: str(row['correct_answer'])==str(row['result_choice']), axis=1)
    df['correctness_answer_first'] = df.apply(lambda row: str(row['correct_answer'])==str(row['result_a_first_choice']), axis=1)
    df['correctness_answer_last'] = df.apply(lambda row: str(row['correct_answer'])==str(row['result_a_last_choice']), axis=1)
    df['correctness_reviewed'] = df.apply(lambda row: str(row['correct_answer'])==str(row['result_review_choice']), axis=1)
    df['consistency'] = df.apply(lambda row: str(row['result_a_first_choice'])==str(row['result_a_last_choice']), axis=1)
    print(df['correctness_raw'].value_counts(True))
    print(df['correctness_answer_first'].value_counts(True))
    print(df['correctness_answer_last'].value_counts(True))
    print(df['correctness_reviewed'].value_counts(True))
    print(df['consistency'].value_counts(True))
