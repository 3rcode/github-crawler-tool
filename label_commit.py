import os
import time
import pandas as pd
from preprocessing import score_by_bleu


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    release_notes_path = os.path.join(ROOT_DIR, 'release_notes.csv')
    commits_path = os.path.join(ROOT_DIR, 'commits.csv')
    release_notes = pd.read_csv(release_notes_path)
    commits = pd.read_csv(commits_path)
    changelog = '\n'.join(release_notes['Release Notes'])
    commits = commits['Messages'][:10]
    start = time.time()
    scores = score_by_bleu(commits, changelog)
    labeled_commits = pd.DataFrame({'Messages': commits, 'Score': scores})
    labeled_commits.to_csv('labeled_commits.csv', index=False)
    print(scores)
    end = time.time()
    print(end - start)
    
