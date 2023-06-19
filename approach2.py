import os
import pandas as pd
import numpy as np
import re
import yaml
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

model = SentenceTransformer('all-mpnet-base-v2')
THRESHOLD = 0.7

linking_statement = [r"(?i)fixes:?", r"(?i)what's changed:?", r"(?i)other changes:?", r"(?i)documentation:?",
                     r"(?i)features:?", r"(?i)new contributors:?", r"(?i)bug fixes:?", r"(?i)changelog:?",
                     r"(?i)notes:?", r"(?i)improvements:?", r"(?i)deprecations:?", r"(?i)minor updates:?",
                     r"(?i)changes:?", r"(?i)fixed:?", r"(?i)maintenance:?", r"(?i)added:?"]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_origin_data(file_path):
    df = pd.read_csv(file_path)
    def merge(row):
        if pd.isna(row['Commit Description']):
            return str(row['Commit Message'])
        else:
            return str(row['Commit Message']) + str(row['Commit Description'])

    commit = df.apply(merge, axis=1)
    label = df['Label']
    return np.asarray(commit), np.asarray(label)


def load_abstract_data(file_path):
    df = pd.read_csv(file_path)
    def merge(row):
        if pd.isna(row['Commit Description Abstract']):
            return str(row['Commit Message Abstract'])
        else:
            return str(row['Commit Message Abstract']) + str(row['Commit Description Abstract'])
    commit = df.apply(merge, axis=1)
    label = df['Label']
    return np.asarray(commit), np.asarray(label)


if __name__ == '__main__':
    repos = []
    data_path = os.path.join(ROOT_DIR, 'data')
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    
    test_cases_path = os.path.join(ROOT_DIR, 'test_cases.yaml')
    test_cases = None
    with open(test_cases_path, 'r') as f:
        test_cases = yaml.load(f, Loader=SafeLoader)

    def approach2(test_name, train_repos, test_repos, _type):
        file_name = 'labeled_commits.csv' if _type == 'origin' else 'labeled_commits_abstract.csv'
        load_data = load_origin_data if _type == 'origin' else load_abstract_data 
        # load changelog sentences database
        all_changelog_sentences = []

        for repo in train_repos:
            path = os.path.join(data_path, repo, 'release_notes.csv')
            df = pd.read_csv(path)
            for release_note in df['Release Note']:
                changelog_sentences = str(release_note).split('\n')
                all_changelog_sentences.extend(changelog_sentences)

        # Remove duplicate sentences and linking statements
        all_changelog_sentences = list(set(all_changelog_sentences))
        all_changelog_sentences = [sentence for sentence in all_changelog_sentences  
                                        if not (any(re.match(pattern, sentence) for pattern in linking_statement) or sentence == '')]
        
        X_test = []
        y_test = []
        # Load commit messages and expected label of commits
        for repo in test_repos:
            path = os.path.join(data_path, repo, file_name)
            commit, label = load_data(path)
            X_test = np.concatenate((X_test, commit), axis=0)
            y_test = np.concatenate((y_test, label), axis=0)

        # Encode all changelog sentences
        print('Start to encode changelog sentences')
        # with open('models/encoded_changelog_sentences.npy', 'r') as f:
        #     encoded_changelog_sentences = np.load(f)
        encoded_changelog_sentences = model.encode(all_changelog_sentences)
        print('Successfully encoded changelog sentences')
        print('Encoded changelog sentences shape:', encoded_changelog_sentences.shape)
        encoded_file = os.path.join(ROOT_DIR, 'models', 'encoded_vectors', f'{test_name}_{_type}')
        with open (encoded_file, 'wb') as f:
            np.save(f, encoded_file)

        # Encode test commit
        print('Start to encode test commit')
        encoded_test_commit = model.encode(X_test)
        print('Successfully encoded test commit')
        print('Encoded test commit shape:', encoded_test_commit.shape)

        # Calculate cosine similarity
        print('Start to calculate cosine similarity')
        cosine_similarities = cosine_similarity(encoded_test_commit, encoded_changelog_sentences)
        scores = np.amax(cosine_similarities, axis=1)
        preds = np.where(scores >= THRESHOLD, 1, 0)

        print('Successfully calculated cosine similarity')
        # Score result
        test_size = len(X_test)
        true_pred = sum([preds[i] == y_test[i] for i in range(test_size)])
        accuracy = true_pred / test_size * 100
        print("Accuracy: %.2f%%" % (accuracy))
        accuracy = str(int(accuracy * 100) / 100) + '%'

        f1 = f1_score(y_test, preds)
        print(f"F1 score: {f1}")
        f1 = str(f1)
        result_path = os.path.join(ROOT_DIR, 'encode_cosine.yaml')
        with open(result_path, 'r') as f:
            result = yaml.safe_load(f)
            if result is None:
                result = {}
            result.update({f'{test_name}_{_type}': {'Num Commits': test_size, 'Accuracy': accuracy, 'F1 score': f1}})
        with open(result_path, 'w') as f:
            yaml.safe_dump(result, f)

    for test_case, test_repos in test_cases.items():
        train_repos = list(set(repos) - set(test_repos))
        approach2(test_name=test_case, train_repos=train_repos, test_repos=test_repos, _type='origin')

