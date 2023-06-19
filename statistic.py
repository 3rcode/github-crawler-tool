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

def summarize_result(approach='naive_bayes'):
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

    def p2f(x):
        return float(x.strip('%')) / 100.0

    def f2p(x):
        return str(round(x * 100.0, 2)) + '%'
    
    result_folder = os.path.join(ROOT_DIR, 'statistic', approach)
    os.mkdir(result_folder)

    accuracy = [p2f(result_origin[i][1]) for i in range(len(result_origin))]
    num_commits = [result_origin[i][3] for i in range(len(result_origin))]
    overall_com = sum(num_commits)
    overall_acc = sum([accuracy[i] * num_commits[i] for i in range(len(result_origin))]) / overall_com
    overall_f1 = sum([float(result_origin[i][2]) for i in range(len(result_origin))]) / len(result_origin)
    summarize = ['Overall', f2p(overall_acc), overall_f1, overall_com]
    result_origin.append(summarize)

    result_origin_path = os.path.join(ROOT_DIR, 'statistic', approach, 'origin.csv')
    df = pd.DataFrame(result_origin, columns=['Test', 'Accuracy', 'F1 Score', 'Num Commits'])
    df.set_index('Test')
    df.to_csv(result_origin_path, index=False)

    accuracy = [p2f(result_abstract[i][1]) for i in range(len(result_abstract))]
    num_commits = [result_abstract[i][3] for i in range(len(result_abstract))]
    overall_com = sum(num_commits)
    overall_acc = sum([accuracy[i] * num_commits[i] for i in range(len(result_abstract))]) / overall_com
    overall_f1 = sum([float(result_abstract[i][2]) for i in range(len(result_abstract))]) / len(result_abstract)
    summarize = ['Overall', f2p(overall_acc), overall_f1, overall_com]
    result_abstract.append(summarize)
    result_abstract_path = os.path.join(ROOT_DIR, 'statistic', approach, 'abstract.csv')
    df = pd.DataFrame(result_abstract, columns=['Test', 'Accuracy', 'F1 Score', 'Num Commits'])
    df.set_index('Test')
    df.to_csv(result_abstract_path, index=False)

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
  

summarize_data()