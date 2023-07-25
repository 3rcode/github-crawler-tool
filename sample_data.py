import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from settings import ROOT_DIR, MODEL
from matplotlib import pyplot as plt
import pathlib


def score_similarity() -> None:
    """ Calculate scores of sample commits """

    sample_path = os.path.join(ROOT_DIR, "data", "sample_data.csv")
    sample_dataset = pd.read_csv(sample_path).astype("str")
    sample_commit_path = os.path.join(ROOT_DIR, "data", "sample_commit")
    if not os.path.exists(sample_commit_path):
        os.mkdir(sample_commit_path)
    print(sample_dataset.head())
    for i in range(len(sample_dataset)):
        _, _, owner, repo, message, description, sha, _, _, _ = sample_dataset.loc[i]
        
        # Load changelog sentences
        folder = f"{owner}_{repo}"
        print(folder)
        c_log_path = os.path.join(ROOT_DIR, "data", folder, "changelog_sentence.csv")
        c_log_sen = pd.read_csv(c_log_path)["Changelog Sentence"].to_numpy().astype("str")

        # Score commit for each changelog sentences
        encoded_mes = MODEL.encode([message], convert_to_numpy=True)
        encoded_c_log_sen = MODEL.encode(c_log_sen, convert_to_numpy=True)
        scores = cosine_similarity(encoded_mes, encoded_c_log_sen)[0]
        scores = np.concatenate((c_log_sen.reshape(-1, 1), scores.reshape(-1, 1)), axis=1)
        c_log_sen = pd.DataFrame(scores, columns=["Attribute", "Value"])
        c_log_sen["Value"] = pd.to_numeric(c_log_sen["Value"])
        c_log_sen = c_log_sen.sort_values(by="Value", ascending=False)
        header = {
            "Label": "",
            "Message": f"{message}",
            "Description": f"{description}",
            "Sha": f"{sha}",
            "Owner": f"{owner}",
            "Repo": f"{repo}",
            "": "",
            "Changelog Sentence": "Score"
        }
        header = np.array(list(header.items()))
        header = pd.DataFrame(header, columns=["Attribute", "Value"])
        record = pd.concat([header, c_log_sen], axis=0)
        print(record.shape)
        record.to_csv(os.path.join(sample_commit_path, f"test_{i + 1}.csv"), index=False)


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
        path = os.path.join(ROOT_DIR, "data", "sample_commit", f"group_{(i - 1) // 20 + 1}", f"test_{i}.csv")
        test = pd.read_csv(path)
        scores = test.loc[8:, "Value"].astype("float64").to_numpy()
        max_score = scores.max()
        commit_scores.append(max_score)
    print("Loaded commit scores")
    accuracy = []
    threshold = np.arange(0, 1, 0.001)
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

    return threshold[accuracy.index(max(accuracy))], max(accuracy)
    

def excel_join():
    for i in [9]:
        input_path = os.path.join(ROOT_DIR, "data", "sample_commit", f"group_{i + 1}")
        output_path = os.path.join(ROOT_DIR, "data", "sample_commit", f"group_{i + 1}.xlsx")
        try:
            with pd.ExcelWriter(output_path) as writer:
                for filename in pathlib.Path(input_path).glob('*.csv'): 
                    df = pd.read_csv(filename)
                    try:
                        df.to_excel(writer, sheet_name=filename.stem, index=False)
                    except:
                        print(filename)
        except Exception as e:  
            print(e)

# print(check_acc_threshold())
# 0.7 73.4375
# 0.71 73.65625