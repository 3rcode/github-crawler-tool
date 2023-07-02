import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.7

if __name__ == '__main__':
    corpus_path = os.path.join(ROOT_DIR, 'data', 'Corpus Repo - Training.csv')
    corpus_repo_training = pd.read_csv(corpus_path)
    num_repo = corpus_repo_training.shape[0]

    for i in range(num_repo):
        owner = corpus_repo_training.loc[i, 'User']
        repo = corpus_repo_training.loc[i, 'Repo name']
        folder_name = f'{owner}_{repo}'
        print(owner, repo)
        try:
            # Load and encode changelog sentences
            changelogs_path = os.path.join(ROOT_DIR, 'data', folder_name, 'changelogs.csv')
            changelogs_sentences = pd.read_csv(changelogs_path)['Changelog Sentence']
            # print(changelogs_sentences.info())

            # Encode changelog sentences
            print('Start to encode changelog sentences')
            encoded_changelog_sentences = model.encode(changelogs_sentences, convert_to_tensor=True)
            print('Successfully encoded changelog sentences')
            
            # Check encoded changelog sentences results
            print('Encoded changelog sentences shape:', encoded_changelog_sentences.shape)

            # Load commit messages
            commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'commits.csv')
            commits = pd.read_csv(commits_path)[['Message', 'Description']]
            # print(commits.info())

            # Drop commit message with same content:
            commits = commits.drop_duplicates(subset='Message', keep='first').reset_index(drop=True)
            # print(commits.info())
            # Encode commit messages
            print('Start to encode commit messages')
            encoded_commit_messages = model.encode(commits['Message'], convert_to_tensor=True)
            print('Successfully encoded commit messages')
            
            # Check encoded commit messages result
            print('Encoded commit messages shape:',encoded_commit_messages.shape)

            # Score commit messages without abstraction
            cosine_similar = cosine_similarity(encoded_commit_messages, encoded_changelog_sentences)
            score = np.amax(cosine_similar, axis=1)
            index = np.argmax(cosine_similar, axis=1)
            cm = np.asarray(commits['Message'])
            ccs = np.array([changelogs_sentences[x] for x in index])
            cd = np.asarray(commits['Description'])
            label = np.where(score >= THRESHOLD, 1, 0)
            df = pd.DataFrame({'Commit Message': cm, 'Score': score, 
                                'Correspond Changelog Sentence': ccs, 
                                'Commit Description': cd, 'Label': label})
            print('Label with normal approach:', df.head())

            # Write label result to the dataset
            labeled_commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'labeled_commits.csv')
            df.to_csv(labeled_commits_path, index=False)
            corpus_repo_training.loc[i, 'Label status'] = 'Done'
        except:
            corpus_repo_training.loc[i, 'Label status'] = 'Error'
        
        corpus_repo_training.to_csv(corpus_path, index=False)
        


        
