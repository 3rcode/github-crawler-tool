import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def collect_statistics():
    repos = []
    data_path = os.path.join(ROOT_DIR, '..', 'data')
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    info = []
    print(len(repos))
    for repo in repos:
        owner, repo_name = repo.split('_')
        path = os.path.join(data_path, repo, 'commits.csv')
        num_commits = len(pd.read_csv(path))
        path = os.path.join(data_path, repo, 'release_notes.csv')
        num_changelogs = len(pd.read_csv(path))
        new_row = [owner, repo_name, num_commits, num_changelogs]
        info.append(new_row)
    
    df = pd.DataFrame(info, columns=['Owner', 'Repo', 'NumCommits', 'NumChangelogs'])
    df.to_csv(os.path.join(ROOT_DIR, 'data_info.csv'), index=False)

collect_statistics()