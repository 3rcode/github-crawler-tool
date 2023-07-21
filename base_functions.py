import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import KFold
from settings import ROOT_DIR
import pygit2
from matplotlib import pyplot as plt

class MyRemoteCallbacks(pygit2.RemoteCallbacks):

    def transfer_progress(self, stats):
        print(f'{stats.indexed_objects}/{stats.total_objects}')


# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     df = df.dropna(subset=["Commit Message"]).reset_index(drop=True)
#     def merge(row):
#         if pd.isna(row["Commit Description"]):
#             return row["Commit Message"]
#         return row["Commit Message"] + "<.> " + row["Commit Description"]
    
#     commit = df.apply(merge, axis=1)
#     label = df["Label"]
#     return np.asarray(commit), np.asarray(label)


# def load_data(test_name, _type="origin", adjust_train_data=False, over_sampling=1, under_sampling=1):
#     path = os.path.join(ROOT_DIR, f"datasets_{_type}", test_name)
#     train_data = pd.read_csv(os.path.join(path, 'train.csv'))
#     test_data = pd.read_csv(os.path.join(path, 'test.csv'))
    
#     if adjust_train_data:
#         train_data_label_1 = train_data[train_data["y_train"] == 1]
#         train_data_label_0 = train_data[train_data["y_train"] == 0]
#         train_data_label_1 = train_data_label_1.sample(frac=over_sampling, replace=True, random_state=0)
#         train_data_label_0 = train_data_label_0.sample(frac=under_sampling, replace=False, random_state=1)
#         train_data = pd.concat([train_data_label_0, train_data_label_1])
#         train_data = train_data.sample(frac=1, random_state=2)

#         X_train = train_data["X_train"]
#         y_train = train_data["y_train"]
#         X_test = test_data["X_test"]
#         y_test = test_data["y_test"]
#     return X_train, y_train, X_test, y_test


# def save_result(result_path, test_case, result):
#     test_name, _type = test_case
#     total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn = result
#     # Need to save result into file
#     _save_result(result_path, {f"{test_name}_{_type}": {"Total test": total_test, 
#                                                         "Precision": precision,
#                                                         "Recall": recall,
#                                                         "F1 score": f1_score,
#                                                         "Accuracy": accuracy,
#                                                         "True negative rate": true_neg_rate,
#                                                         "True positive": tp,
#                                                         "True negative": tn,
#                                                         "False positive": fp,
#                                                         "False negative": fn
#                                                         }})


# def _save_result(path, content):
#     with open(path, 'r') as f:
#         result = yaml.safe_load(f)
#         if result is None:
#             result = {}
#         result.update(content)
#     with open(path, 'w') as f:
#         yaml.safe_dump(result, f)


# def analyze_result(path, test_case, commits, prediction, Y):
#     test_name, _type = test_case
#     tp = sum([pred == y and pred == 1 for pred, y in zip(prediction, Y)])
#     tn = sum([pred == y and pred == 0 for pred, y in zip(prediction, Y)])
#     fp = sum([pred != y and pred == 1 for pred, y in zip(prediction, Y)])
#     fn = sum([pred != y and pred == 0 for pred, y in zip(prediction, Y)])

#     # false_case = [index for index in range(len(Y)) if prediction[index] != Y[index]]
#     # true_case = [index for index in range(len(Y)) if prediction[index] == Y[index]]
#     # true_positive = [commits[index] for _, index in enumerate(true_case) if prediction[index] == 1]
#     # true_negative = [commits[index] for _, index in enumerate(true_case) if prediction[index] == 0]
#     # false_positive = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 1]
#     # false_negative = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 0]
#     # _save_result(path, {f'{test_name}_{_type}': {'False Positive': false_positive, 
#     #                                              'False Negative': false_negative}})
#     # tp = len(true_positive)
#     # tn = len(true_negative)
#     # fp = len(false_positive)
#     # fn = len(false_negative)
    
#     total_test = len(Y)
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     true_neg_rate = tn / (tn + fp)
#     f1_score = 2 * precision * recall / (precision + recall)
#     accuracy = (tp + tn) / (tp + tn + fp + fn)

    
#     return total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn


# def find_commit(commit, _type="origin"):
#     file = "labeled_commits.csv" if _type == "origin" else "labeled_commits_abstract.csv"
#     data_path = os.path.join(ROOT_DIR, "data")
#     repos = []
#     for subdir, dirs, files in os.walk(data_path):
#         repos.extend(dirs)
#     results = []

#     for repo in repos:
#         path = os.path.join(data_path, repo, file)
#         df = pd.read_csv(path)
#         row = df.loc[(df["Commit Message"] == commit) | (df["Commit Message"] + "<.> " + df["Commit Description"].astype("str") == commit)] 
#         if not row.empty:
#             mes, score, ccs, des, label = np.squeeze(row.to_numpy()).astype(str)
#             results.append("Repo {}:\nMessage:{}\nScore:{}\nCorrespond Changelog Sentence:{}\nCommit Description:{}\nLabel:{}\n\n".format(repo, mes, score, ccs, des, label))
#     return results


# def show_result(result):
#     total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn = result
#     print("Total test:", total_test)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 score:", f1_score)
#     print("Accuracy:", accuracy)
#     print("True negative rate:", true_neg_rate)
#     print("True positive:", tp)
#     print("True negative:", tn)
#     print("False positive:", fp)
#     print("False negative", fn)


# def k_fold_splitter(_type="origin"):
#     # Load data in all repositories
#     file_name = "labeled_commits.csv" if _type == "origin" else "labeled_commits_abstract.csv"
#     data_path = os.path.join(ROOT_DIR, "data")
#     repos = []
#     for subdir, dirs, files in os.walk(data_path):
#         repos.extend(dirs)
#     dfs = []
#     for repo in repos:
#         path = os.path.join(data_path, repo, file_name)
#         df = pd.read_csv(path)
#         dfs.append(df)

#     # Merge all repositories data into one dataframe 
#     all_data = pd.concat(dfs)
#     all_data = all_data.dropna(subset=["Commit Message"]).reset_index(drop=True)

#     # Remove all duplicates of commit message, if that commit message has any occurance 
#     # with label 1 then label that commit message 1 
#     all_data = all_data.sort_values("Label").drop_duplicates("Commit Message", keep="last").reset_index(drop=True)
    
#     # Shuffle data after sort:
#     all_data = all_data.sample(frac=1, axis=0, random_state=5)
    
#     # Get commit messages and commit descriptions as features of models and label is label
#     def merge(row):
#         if pd.isna(row["Commit Description"]):
#             return row["Commit Message"]
#         else:
#             return row["Commit Message"] + "<.> " + row["Commit Description"]

#     commit = all_data.apply(merge, axis=1)
#     label = all_data["Label"]
#     data = np.column_stack((commit, label))

#     # Create folder to save datasets
#     folder = f"datasets_{_type}"
#     folder_path = os.path.join(ROOT_DIR, folder)
#     if not os.path.exists(folder_path):
#         os.mkdir(folder_path)

#     # Use K-Fold technique to separate data into 10 (train, test) datasets and save it into folder
#     kf = KFold(n_splits=10, shuffle=True, random_state=10)
#     for i, (train_index, test_index) in enumerate(kf.split(data)):
#         train_data = pd.DataFrame(data[train_index], columns=["X_train", "y_train"])
#         test_data = pd.DataFrame(data[test_index], columns=["X_test", "y_test"])
#         print(train_data.head())
#         path = os.path.join(folder_path, f"test_{i + 1}")
#         if not os.path.exists(path):
#             os.mkdir(path)
#         train_data.to_csv(os.path.join(path, "train.csv"), index=False)
#         test_data.to_csv(os.path.join(path, "test.csv"), index=False)


def clone_repos() -> None:
    """ Clone all repositories in "Repos.csv" file """
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    num_repo = len(repos)

    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        path = os.path.join(ROOT_DIR, "..", "data", f"{owner}_{repo}")
        pygit2.clone_repository(f"https://github.com/{owner}/{repo}", path, callbacks=MyRemoteCallbacks())


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
        path = os.path.join(ROOT_DIR, "data", f"{owner}_{repo}", "commits.csv")
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
    all_data["Index"] = [idx + 1 for idx in all_data.index]
    print(all_data.head())
    all_data.to_csv(os.path.join(ROOT_DIR, "data", "all_data.csv"), index=False)


def sampling_dataset() -> None:
    """ Sample commits from all commits """

    path = os.path.join(ROOT_DIR, "data", "all_data.csv")
    sample_path = os.path.join(ROOT_DIR, "data", "sample_data.csv")
    all_data = pd.read_csv(path)
    print(all_data.info())
    sample_dataset = all_data.sample(n=384)
    sample_dataset.to_csv(sample_path, index=False)

def check_acc_threshold() -> float:
    # human_label_path = os.path.join(ROOT_DIR, "data", "human_label.csv")
    # human_label = pd.read_csv(human_label_path)["Label"].to_numpy()
    commit_scores = []
    for i in range(1, 385):
        path = os.path.join(ROOT_DIR, "data", "sample_commit", f"test_{i}.csv")
        test = pd.read_csv(path)
        print(test.head())
        scores = test.loc[8:, "Value"].astype("float64").to_numpy()
        max_score = scores.max()
        commit_scores.append(max_score)
    threshold = range(0, 1, 0.01)
    for i in threshold:
        pass





check_acc_threshold()




