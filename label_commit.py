import time
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = 'electron_electron'


if __name__ == '__main__':
    # Initial
    # Load and encode changelog sentences
    release_notes_path = os.path.join(ROOT_DIR, REPO, 'release_notes.csv')
    release_notes = pd.read_csv(release_notes_path)
    changelog = '\n'.join(release_notes['Release Note'])
    changelog_sentences = list(set(changelog.split('\n')))
    print('Start to encode changelog sentences')
    encoded_changelog_sentences = model.encode(changelog_sentences)
    print('Successfully encoded changelog sentences')
    print(encoded_changelog_sentences.shape)

    # Load and encode commit messages
    commits_path = os.path.join(ROOT_DIR, REPO, 'commits.csv')
    commits = pd.read_csv(commits_path)
    commit_messages = commits['Message']
    commit_description = commits['Description']
    print('Start to encode commit messages')
    encoded_commit_messages = model.encode(commit_messages)
    print('Successfully encoded commit messages')

    # Score commit messages
    start = time.time()
    results = []
    for index in range(len(commits)):
        result = list(cosine_similarity([encoded_commit_messages[index]], encoded_changelog_sentences)[0])
        score = max(result)
        index_max = result.index(score)
        cm = commit_messages[index]
        ccs = changelog_sentences[index_max]
        cd = commit_description[index]
        print('{0:7.5f}\t\t{1}'.format(score, cm))
        new_record = [index + 1, cm, score, ccs, cd]
        results.append(new_record)

    df = pd.DataFrame(results, 
                      columns=['Index', 'Commit Message', 'Score', 'Correspond Changelog Sentence', 'Commit Description'])
    labeled_commits_path = os.path.join(ROOT_DIR, REPO, 'labeled_commits.csv')
    df.to_csv(labeled_commits_path, index=False)    
    end = time.time()
    print(end - start)
