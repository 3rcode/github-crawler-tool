import os
import pandas as pd
import numpy as np
import re
import yaml
from yaml.loader import SafeLoader

linking_statement = [r"(?i)fixes:?", r"(?i)what's changed:?", r"(?i)other changes:?", r"(?i)documentation:?",
                     r"(?i)features:?", r"(?i)new contributors:?", r"(?i)bug fixes:?", r"(?i)changelog:?",
                     r"(?i)notes:?", r"(?i)improvements:?", r"(?i)deprecations:?", r"(?i)minor updates:?",
                     r"(?i)changes:?", r"(?i)fixed:?", r"(?i)maintenance:?", r"(?i)added:?"]

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def summarize_result(approach):
    result_path = os.path.join(ROOT_DIR, f'{approach}.yaml')
    result = None
    with open(result_path, 'r') as f:
        result = yaml.load(f, Loader=SafeLoader)

    result_origin = []
    result_abstract = []

    for key, value in result.items():
        test_case, _type = key.split('_')
        
        if _type == 'origin':
            result_origin.append([test_case, *value.values()])
        else:
            result_abstract.append([test_case, *value.values()])
    
    origin = pd.DataFrame(result_origin, columns=['Test case', 'Accuracy', 'F1 score',  'Precision', 'Recall', 'Total test',  'True negative rate'])
    abstract = pd.DataFrame(result_abstract, columns=['Test case', 'Accuracy', 'F1 score',  'Precision', 'Recall', 'Total test',  'True negative rate'])
    def _summary(df):
        np_df = df.to_numpy()
        overall_tt = np.sum(np_df[:, 5], axis=0)
        overall_a = np.sum(np_df[:, 1] * np_df[:, 5], axis=0) / overall_tt
        overall_f = np.sum(np_df[:, 2] * np_df[:, 5], axis=0) / overall_tt
        overall_p = np.sum(np_df[:, 3] * np_df[:, 5], axis=0) / overall_tt
        overall_r = np.sum(np_df[:, 4] * np_df[:, 5], axis=0) / overall_tt
        overall_tnr = np.sum(np_df[:, 6] * np_df[:, 5], axis=0) / overall_tt
        return ['Overall', overall_a, overall_f, overall_p, overall_r, overall_tt, overall_tnr]
    
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
    data_path = os.path.join(ROOT_DIR, 'data')
    repos = []
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)

    def num_release_notes_sentences(release_notes):
        result = []
        for col in ['Release Note', 'Release Note Abstract']:
            all_release_note_sentences = []
            for release_note in release_notes[col]:
                release_note_sentences = str(release_note).split('\n')
                all_release_note_sentences.extend(release_note_sentences)
            
            # Remove duplicate sentences and linking statements
            all_release_note_sentences = list(set(all_release_note_sentences))
            all_release_note_sentences = [sentence for sentence in all_release_note_sentences  
                                            if not (any(re.match(pattern, sentence) for pattern in linking_statement) or sentence == '')]
            result.append(len(all_release_note_sentences))
        return result
    
    raw_data = []
    origin_data = []
    abstract_data = []
    
    for repo in repos:
        repo_dir = os.path.join(data_path, repo)
        release_notes = os.path.join(repo_dir, 'release_notes.csv')
        commits = os.path.join(repo_dir, 'commits.csv')
        labeled_commits_origin = os.path.join(repo_dir, 'labeled_commits.csv')
        labeled_commits_abstract = os.path.join(repo_dir, 'labeled_commits_abstract.csv')

        # Load raw data
        release_notes = pd.read_csv(release_notes)
        commits = pd.read_csv(commits)
        num_release_notes = release_notes.shape[0]
        num_raw_commits = commits.shape[0]

        raw_data.append([repo, num_raw_commits, num_release_notes])

        # Load processed data
        labeled_commits_origin = pd.read_csv(labeled_commits_origin)
        labeled_commits_abstract = pd.read_csv(labeled_commits_abstract)

        num_release_note_sentences_origin, num_release_note_sentences_abstract = num_release_notes_sentences(release_notes)
        num_processed_commit_origin = labeled_commits_origin.shape[0]
        num_commit_label_origin = labeled_commits_origin['Label'].value_counts()

        num_processed_commit_abstract = labeled_commits_abstract.shape[0]
        num_commit_label_abstract = labeled_commits_abstract['Label'].value_counts()
        print(num_commit_label_abstract)
        
        origin_data.append([repo, num_processed_commit_origin, num_release_note_sentences_origin,
                            num_commit_label_origin[0], num_commit_label_origin[1]])
        
        abstract_data.append([repo, num_processed_commit_abstract, num_release_note_sentences_abstract,
                            num_commit_label_abstract[0], num_commit_label_abstract[1]])
        
    data_info_folder = os.path.join(ROOT_DIR, 'statistic', 'data_info')
    os.mkdir(data_info_folder)
    raw_data = pd.DataFrame(raw_data, columns=['Repo', 'Num Raw Commit', 'Num Release Note'])
    raw_data.set_index('Repo')
    raw_data.to_csv(os.path.join(data_info_folder, 'raw_data.csv'))

    origin_data = pd.DataFrame(origin_data, columns=['Repo', 'Num Commit', 'Num Release Note Sentece',
                                                    'Num Label 0', 'Num Label 1'])
    origin_data.set_index('Repo')
    origin_data.to_csv(os.path.join(data_info_folder, 'origin_data.csv'))

    abstract_data = pd.DataFrame(abstract_data, columns=['Repo', 'Num Commit', 'Num Release Note Sentece',
                                                    'Num Label 0', 'Num Label 1'])
    abstract_data.set_index('Repo')
    abstract_data.to_csv(os.path.join(data_info_folder, 'abstract_data.csv'))
  

summarize_result(approach='encode_cosine')