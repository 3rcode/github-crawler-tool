import os
import pandas as pd
from typing import Callable 
from settings import ROOT_DIR


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


def traverse_repos(func: Callable[[str, str], None]) -> None:
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
            




