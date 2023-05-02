import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def from_score_to_class():
    repos = []
    data_path = os.path.join(ROOT_DIR, 'data')
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)

    cnt = 0
    for repo in repos:
        path = os.path.join(data_path, repo, 'labeled_commits.csv')
        df = pd.read_csv(path)
        df['Class'] = df['Score'].apply(lambda score: 1 if score > 0.7 else 0)
        df.to_csv(path, index=False)


from_score_to_class()