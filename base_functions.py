import pandas as pd
import numpy as np
import os
import random
import yaml
from sklearn.model_selection import KFold

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(file_path):
    df = pd.read_csv(file_path)
    def merge(row):
        if pd.isna(row['Commit Description']):
            return str(row['Commit Message'])
        else:
            return str(row['Commit Message']) + "\n" + str(row['Commit Description'])

    commit = df.apply(merge, axis=1)
    label = df['Label']
    return np.asarray(commit), np.asarray(label)

def load_data(test_name, _type, adjust_train_data=False, over_sampling=1, under_sampling=1):
    path = os.path.join(ROOT_DIR, 'datasets_'+_type, test_name)
    train_data = pd.read_csv(os.path.join(path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(path, 'test.csv'))
    
    if adjust_train_data:
        train_data_label_1 = train_data[train_data['y_train'] == 1]
        train_data_label_0 = train_data[train_data['y_train'] == 0]
        train_data_label_1 = train_data_label_1.sample(frac=over_sampling, replace=True, random_state=0)
        train_data_label_0 = train_data_label_0.sample(frac=under_sampling, replace=False, random_state=1)
        train_data = pd.concat([train_data_label_0, train_data_label_1])
        train_data = train_data.sample(frac=1, random_state=2)

    X_train = train_data['X_train'].astype('str')
    y_train = train_data['y_train'].astype('bool')
    X_test = test_data['X_test'].astype('str')
    y_test = test_data['y_test'].astype('bool')
    return X_train, y_train, X_test, y_test

def save_result(result_path, test_case, result):
    test_name, _type = test_case
    total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn = result
    # Need to save result into file
    _save_result(result_path, {f'{test_name}_{_type}': {'Total test': total_test, 
                                                        'Precision': precision,
                                                        'Recall': recall,
                                                        'F1 score': f1_score,
                                                        'Accuracy': accuracy,
                                                        'True negative rate': true_neg_rate,
                                                        'True positive': tp,
                                                        'True negative': tn,
                                                        'False positive': fp,
                                                        'False negative': fn
                                                        }})
    
def _save_result(path, content):
    with open(path, 'r') as f:
        result = yaml.safe_load(f)
        if result is None:
            result = {}
        result.update(content)
    with open(path, 'w') as f:
        yaml.safe_dump(result, f)

def analyze_result(path, test_case, commits, prediction, Y):
    test_name, _type = test_case
    false_case = [index for index in range(len(Y)) if prediction[index] != Y[index]]
    true_case = [index for index in range(len(Y)) if prediction[index] == Y[index]]
    true_positive = [commits[index] for _, index in enumerate(true_case) if prediction[index] == 1]
    true_negative = [commits[index] for _, index in enumerate(true_case) if prediction[index] == 0]
    false_positive = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 1]
    false_negative = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 0]
    
    tp = len(true_positive)
    tn = len(true_negative)
    fp = len(false_positive)
    fn = len(false_negative)
    
    total_test = len(Y)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    true_neg_rate = tn / (tn + fp)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # _save_result(path, {f'{test_name}_{_type}': {'False Positive': false_positive, 
    #                                              'False Negative': false_negative}})
    return total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn

def find_commit(commit, _type):
    file = 'labeled_commits.csv' if _type == 'origin' else 'labeled_commits_abstract.csv'
    data_path = os.path.join(ROOT_DIR, 'data')
    repos = []
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    results = []

    for repo in repos:
        path = os.path.join(data_path, repo, file)
        df = pd.read_csv(path)
        row = df.loc[(df['Commit Message'].astype(str) == commit) | (df['Commit Message'].astype(str) + '\n' + df['Commit Description'].astype(str) == commit)] 
        if not row.empty:
            results.append('Repo ({}): {}'.format(repo, '||'.join(np.squeeze(row.to_numpy()).astype(str))))
    return results

def show_result(result):
    total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn = result
    print("Total test:", total_test)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1_score)
    print("Accuracy:", accuracy)
    print("True negative rate:", true_neg_rate)
    print("True positive:", tp)
    print("True negative:", tn)
    print("False positive:", fp)
    print("False negative", fn)

def k_fold_splitter(_type):
    # Load data in all repositories
    file_name = 'labeled_commits.csv' if _type == 'origin' else 'labeled_commits_abstract.csv'
    data_path = os.path.join(ROOT_DIR, 'data')
    repos = []
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    dfs = []
    for repo in repos:
        path = os.path.join(data_path, repo, file_name)
        df = pd.read_csv(path)
        dfs.append(df)

    # Merge all repositories data into one dataframe 
    all_data = pd.concat(dfs)

    # Remove all duplicates of commit message, if that commit message has any occurance 
    # with label 1 then label that commit message 1 
    all_data = all_data.sort_values('Label').drop_duplicates('Commit Message', keep='last')
    
    # Shuffle data after sort:
    all_data = all_data.sample(frac=1, axis=0, random_state=5)
    
    # Get commit messages and commit descriptions as features of models and label is label
    def merge(row):
        if pd.isna(row['Commit Description']):
            return str(row['Commit Message'])
        else:
            return str(row['Commit Message']) + "\n" + str(row['Commit Description'])

    commit = all_data.apply(merge, axis=1)
    label = all_data['Label']
    data = np.column_stack((commit, label))

    # Create folder to save datasets
    folder = 'datasets_' + _type
    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Use K-Fold technique to separate data into 10 (train, test) datasets and save it into folder
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train_data = pd.DataFrame(data[train_index], columns=['X_train', 'y_train'])
        test_data = pd.DataFrame(data[test_index], columns=['X_test', 'y_test'])
        print(train_data.head())
        path = os.path.join(folder_path, f'test_{i + 1}')
        if not os.path.exists(path):
            os.mkdir(path)
        train_data.to_csv(os.path.join(path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test.csv'), index=False)

if __name__ == '__main__':
    print(find_commit('Backport #5871 for v3.4.x: Convert StaticFile liquid representation to a Drop & add front matter defaults support to StaticFiles #5940', 'origin'))
