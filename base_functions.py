import os
import pandas as pd
from typing import Callable 
from settings import ROOT_DIR



def traverse_repos(func: Callable[[str, str], None]) -> None:
    """ This function do func in range of all repositories in Repos.csv file"""

    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    # Check repos
    print(repos.shape)
    print(repos.head())
    num_repo = len(repos)
    error_log = open("error_log.txt", "a+")
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        try:
            func(owner, repo)
        except Exception as e:
            error_log.writelines(f"Repo {owner}/{repo} encounter error: {e} in function {func.__name__}")
    error_log.close()
            




