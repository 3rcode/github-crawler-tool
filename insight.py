import os
import pandas as pd
from settings import ROOT_DIR

def check_bot():
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    num_repo = len(repos)
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder = f"{owner}_{repo}"
        path = os.path.join(ROOT_DIR, "data", folder, "commit.csv")
        commit_info = pd.read_csv(path)
        print(commit_info.head())
        break

check_bot()