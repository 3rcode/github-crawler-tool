import os
import pandas as pd
import numpy as np
# import yaml
# from yaml.loader import SafeLoader
# from collections import defaultdict
from settings import ROOT_DIR

# def summarize_result(approach):
#     result_path = os.path.join(ROOT_DIR, f'{approach}.yml')
#     result = None
#     with open(result_path, 'r') as f:
#         result = yaml.load(f, Loader=SafeLoader)

#     result_origin = []
#     result_abstract = []

#     for key, value in result.items():
#         print(key)
#         tmp, test_case, _type = key.split('_')
#         test_case = tmp + '_' + test_case
        
#         if _type == 'origin':
#             result_origin.append([test_case, *value.values()])
#         else:
#             result_abstract.append([test_case, *value.values()])
    
#     origin = pd.DataFrame(result_origin, columns=['Test case', 'Accuracy', 'F1 score',  'Precision', 'Recall', 'Total test',  'True negative rate', 'True positive', 'True negative', 'False positive', 'False negative'])
#     abstract = pd.DataFrame(result_abstract, columns=['Test case', 'Accuracy', 'F1 score',  'Precision', 'Recall', 'Total test',  'True negative rate', 'True positive', 'True negative', 'False positive', 'False negative'])
#     def _summary(df):
#         np_df = df.to_numpy()
#         overall_tt = np.sum(np_df[:, 5], axis=0)
#         overall_a = np.sum(np_df[:, 1] * np_df[:, 5], axis=0) / overall_tt
#         overall_f = np.sum(np_df[:, 2] * np_df[:, 5], axis=0) / overall_tt
#         overall_p = np.sum(np_df[:, 3] * np_df[:, 5], axis=0) / overall_tt
#         overall_r = np.sum(np_df[:, 4] * np_df[:, 5], axis=0) / overall_tt
#         overall_tnr = np.sum(np_df[:, 6] * np_df[:, 5], axis=0) / overall_tt
#         overall_tp = np.sum(np_df[:, 7] * np_df[:, 5], axis=0) / overall_tt
#         overall_tn = np.sum(np_df[:, 8] * np_df[:, 5], axis=0) / overall_tt
#         overall_fp = np.sum(np_df[:, 9] * np_df[:, 5], axis=0) / overall_tt
#         overall_fn = np.sum(np_df[:, 10] * np_df[:, 5], axis=0) / overall_tt
#         return ['Overall', overall_a, overall_f, overall_p, overall_r, overall_tt, overall_tnr, overall_tp, overall_tn, overall_fp, overall_fn]
    
#     origin.loc[len(origin)] = _summary(origin)
#     origin.set_index('Test case')
#     abstract.loc[len(abstract)] = _summary(abstract)
#     abstract.set_index('Test case')
#     result_folder = os.path.join(ROOT_DIR, 'statistic', f'{approach}')
#     os.mkdir(result_folder)
#     path_to_origin = os.path.join(result_folder, 'origin.csv')
#     origin.to_csv(path_to_origin)
#     path_to_abstract = os.path.join(result_folder, 'abstract.csv')
#     abstract.to_csv(path_to_abstract)


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

summarize_data()
  