import pandas as pd
import numpy as np
import os
import random
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def load_data(file_path):
    df = pd.read_csv(file_path)
    def merge(row):
        if pd.isna(row['Commit Description']):
            return str(row['Commit Message'])
        else:
            return str(row['Commit Message']) + str(row['Commit Description'])

    commit = df.apply(merge, axis=1)
    label = df['Label']
    return np.asarray(commit), np.asarray(label)


def save_result(result_path, test_case, result):
    test_name, _type = test_case
    test_size, accuracy, f1 = result
    _save_result(result_path, {f'{test_name}_{_type}': {'Num Commits': test_size, 'Accuracy': accuracy, 'F1 score': f1}})
    
    
def _save_result(path, content):
    with open(path, 'r') as f:
        result = yaml.safe_load(f)
        if result is None:
            result = {}
        result.update(content)
    with open(path, 'w') as f:
        yaml.safe_dump(result, f)


def sample_wrong_cases(path, test_case, commits, prediction, Y):
    test_name, _type = test_case
    false_case = [index for index in range(len(Y)) if prediction[index] != Y[index]]
    false_positive = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 1]
    false_negative = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 0]
    sample_fp = random.choices(false_positive, k=5)
    sample_fp_info = {commit: find_commit(commit) for commit in sample_fp}
    sample_fn = random.choices(false_negative, k=5)
    sample_fn_info = {commit: find_commit(commit) for commit in sample_fn}
    _save_result(path, {f'{test_name}_{_type}': {'False Positive': sample_fp_info, 
                                                 'False Negative': sample_fn_info}})

    

def find_commit(commit):
    data_path = os.path.join(ROOT_DIR, 'data')
    repos = []
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    results = []
    for repo in repos:
        path = os.path.join(data_path, repo, 'labeled_commits.csv')
        df = pd.read_csv(path)
        row = df.loc[df['Commit Message'] == commit] 
        if not row.empty:
            results.append('{}: {}'.format(repo, '||'.join(np.squeeze(row.to_numpy()).astype(str))))
    return results

find_commit('Hello World')

