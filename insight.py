import os
import pandas as pd
from settings import ROOT_DIR
import matplotlib.pyplot as plt
import numpy as np
import re


def join_dataset() -> None:
    """ Join commits of all repositories into one file """

    # Load all repositories commits into dataframes
    repo_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repo_path)
    num_repo = len(repos)
    dfs = []
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        path = os.path.join(ROOT_DIR, "data", f"{owner}_{repo}", "commit.csv")
        df = pd.read_csv(path)
        dfs.append(df)

    # Merge all repositories commits into one dataframe 
    all_data = pd.concat(dfs)
    # Remove null commit has no message
    all_data = all_data.dropna(subset=["Message"]).reset_index(drop=True)
    # Remove duplicate commits has same message
    all_data = all_data.drop_duplicates(subset=["Message"]).reset_index(drop=True)
    # Shuffle commits
    all_data = all_data.sample(frac=1, axis=0).reset_index(drop=True)
    # Number commits
    print(all_data.head())
    all_data.to_csv(os.path.join(ROOT_DIR, "data", "all_data.csv"), index=False)


def sampling_dataset() -> None:
    """ Sample commits from all commits """

    path = os.path.join(ROOT_DIR, "data", "all_data.csv")
    sample_path = os.path.join(ROOT_DIR, "data", "sample_data.csv")
    all_data = pd.read_csv(path)
    sample_dataset = all_data.sample(n=384)
    sample_dataset = sample_dataset.drop(columns="Index")
    sample_dataset = sample_dataset.reset_index()
    sample_dataset.index.name = "Index"
    print(sample_dataset.info())
    sample_dataset.to_csv(sample_path)


def check_acc_threshold() -> float:
    human_label_path = os.path.join(ROOT_DIR, "data", "human_label.csv")
    human_label = pd.read_csv(human_label_path)["Label"].astype("float64").to_numpy()
    commit_scores = []
    for i in range(1, 385):
        path = os.path.join(ROOT_DIR, "data", "sample_commit", f"test_{i}.csv")
        test = pd.read_csv(path)
        scores = test.loc[8:, "Value"].astype("float64").to_numpy()
        max_score = scores.max()
        commit_scores.append(max_score)
    print("Loaded commit scores")
    accuracy = []
    threshold = np.arange(0, 1, 0.01)
    for i in threshold:
        cnt = 0
        for k in range(384):
            if ((commit_scores[k] >= i and human_label[k] == 1) or
                (commit_scores[k] < i and human_label[k] == 0)):
                cnt += 1    
        accuracy.append(cnt / 384 * 100)
    
    print(accuracy)
    plt.plot(threshold, accuracy, color="tab:blue", linestyle="solid")
    plt.title("Label accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.savefig("label_accuracy.png")

    return threshold[accuracy.index(max(accuracy))]
    


def summarize_data():
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    data = []
    for i in range(len(repos)):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder = f"{owner}_{repo}"
        changelog_sen_path = os.path.join(ROOT_DIR, "data", folder, "changelog_sentence.csv")
        commit_path = os.path.join(ROOT_DIR, "data", folder, "commit.csv")
        

        # Load raw data
        changelog_sen_df = pd.read_csv(changelog_sen_path)
        commit_df = pd.read_csv(commit_path)
        num_changelog_sen = len(changelog_sen_df)
        num_commit = len(commit_df)

        data.append([repo, num_commit, num_changelog_sen])        
        

    data = pd.DataFrame(data, columns=["Repo", "Num Commit", "Num Changelog Sentence"])
    data.set_index("Repo")
    data.to_csv("data_info.csv", index=False)


def check_bot():
    path = os.path.join(ROOT_DIR, "data", "all_data.csv")
    all_data = pd.read_csv(path)
    print(all_data.info())
    pattern_author_bot = r".*\[bot\].*"
    num_author_bot = sum([1 if re.search(pattern_author_bot, author) else 0 for author in all_data.loc[:, "Author"]])
    num_github_committer = len(all_data[all_data["Committer"] == "GitHub <noreply@github.com>"])
    print(num_author_bot)
    print(num_github_committer)

def check_commit