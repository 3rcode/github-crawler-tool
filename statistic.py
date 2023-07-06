import os
import pandas as pd
import numpy as np
import yaml
from yaml.loader import SafeLoader
from collections import defaultdict
from settings import ROOT_DIR

def summarize_result(approach):
    result_path = os.path.join(ROOT_DIR, f'{approach}.yml')
    result = None
    with open(result_path, 'r') as f:
        result = yaml.load(f, Loader=SafeLoader)

    result_origin = []
    result_abstract = []

    for key, value in result.items():
        print(key)
        tmp, test_case, _type = key.split('_')
        test_case = tmp + '_' + test_case
        
        if _type == 'origin':
            result_origin.append([test_case, *value.values()])
        else:
            result_abstract.append([test_case, *value.values()])
    
    origin = pd.DataFrame(result_origin, columns=['Test case', 'Accuracy', 'F1 score',  'Precision', 'Recall', 'Total test',  'True negative rate', 'True positive', 'True negative', 'False positive', 'False negative'])
    abstract = pd.DataFrame(result_abstract, columns=['Test case', 'Accuracy', 'F1 score',  'Precision', 'Recall', 'Total test',  'True negative rate', 'True positive', 'True negative', 'False positive', 'False negative'])
    def _summary(df):
        np_df = df.to_numpy()
        overall_tt = np.sum(np_df[:, 5], axis=0)
        overall_a = np.sum(np_df[:, 1] * np_df[:, 5], axis=0) / overall_tt
        overall_f = np.sum(np_df[:, 2] * np_df[:, 5], axis=0) / overall_tt
        overall_p = np.sum(np_df[:, 3] * np_df[:, 5], axis=0) / overall_tt
        overall_r = np.sum(np_df[:, 4] * np_df[:, 5], axis=0) / overall_tt
        overall_tnr = np.sum(np_df[:, 6] * np_df[:, 5], axis=0) / overall_tt
        overall_tp = np.sum(np_df[:, 7] * np_df[:, 5], axis=0) / overall_tt
        overall_tn = np.sum(np_df[:, 8] * np_df[:, 5], axis=0) / overall_tt
        overall_fp = np.sum(np_df[:, 9] * np_df[:, 5], axis=0) / overall_tt
        overall_fn = np.sum(np_df[:, 10] * np_df[:, 5], axis=0) / overall_tt
        return ['Overall', overall_a, overall_f, overall_p, overall_r, overall_tt, overall_tnr, overall_tp, overall_tn, overall_fp, overall_fn]
    
    origin.loc[len(origin)] = _summary(origin)
    origin.set_index('Test case')
    abstract.loc[len(abstract)] = _summary(abstract)
    abstract.set_index('Test case')
    result_folder = os.path.join(ROOT_DIR, 'statistic', f'{approach}')
    os.mkdir(result_folder)
    path_to_origin = os.path.join(result_folder, 'origin.csv')
    origin.to_csv(path_to_origin)
    path_to_abstract = os.path.join(result_folder, 'abstract.csv')
    abstract.to_csv(path_to_abstract)



def summarize_data():
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    data_path = os.path.join(ROOT_DIR, "data")
    repos = pd.read_csv(repos_path)
    repos = repos[(repos["Crawl status"] == "Done") 
                                & (repos["Label status"] == "Done")]

    repos = (repos.loc[:, "Owner"] + '_' + repos.loc[:, "Repo"]).tolist()
    data = []
    for repo in repos:
        repo_dir = os.path.join(data_path, repo)
        changelogs_path = os.path.join(repo_dir, "changelogs.csv")
        commits_path = os.path.join(repo_dir, "commits.csv")
        labelled_commits_path= os.path.join(repo_dir, 'labelled_commits.csv')

        # Load raw data
        changelogs = pd.read_csv(changelogs_path)
        commits = pd.read_csv(commits_path)
        num_changelog_sentences = len(changelogs)
        num_commits = len(commits)

        # Load processed data
        labelled_commits = pd.read_csv(labelled_commits_path)

        num_commit_labels = defaultdict(int, labelled_commits["Label"].value_counts())

        data.append([repo, num_commits, num_changelog_sentences, 
                     num_commit_labels[0], num_commit_labels[1]])        
        
        
    data_info_path = os.path.join(ROOT_DIR, "statistic", "data_info.csv")
    data = pd.DataFrame(data, columns=["Repo", "Num Commit", "Num Changelog Sentence",
                                       "Num Label 0", "Num Label 1"])
    data.set_index("Repo")
    data.to_csv(data_info_path, index=False)
  