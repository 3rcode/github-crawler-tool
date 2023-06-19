import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')
linking_statement = [r"(?i)fixes:?", r"(?i)what's changed:?", r"(?i)other changes:?", r"(?i)documentation:?",
                     r"(?i)features:?", r"(?i)new contributors:?", r"(?i)bu g fixes:?", r"(?i)changelog:?",
                     r"(?i)notes:?", r"(?i)improvements:?", r"(?i)deprecations:?", r"(?i)minor updates:?",
                     r"(?i)changes:?", r"(?i)fixed:?", r"(?i)maintenance:?", r"(?i)added:?"]

THRESHOLD = 0.7

if __name__ == '__main__':
    print('Something')
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(ROOT_DIR, 'data', 'Corpus Repo - Training.csv')
    corpus_repo_training = pd.read_csv(corpus_path)
    # Check corpus repo training
    print(corpus_repo_training.shape)
    print(corpus_repo_training.head())
    
    num_repo = corpus_repo_training.shape[0]
    for i in range(num_repo):
        label_status = corpus_repo_training.loc[i, 'Label status']
        if label_status != 'Done':
            owner = corpus_repo_training.loc[i, 'User']
            repo = corpus_repo_training.loc[i, 'Repo name']
            folder_name = f'{owner}_{repo}'
            print(owner, repo)

            # Load and encode changelog sentences
            release_notes_path = os.path.join(ROOT_DIR, 'data', folder_name, 'release_notes.csv')
            release_notes = pd.read_csv(release_notes_path)
            changelog = '\n'.join(map(str, release_notes['Release Note']))
            changelog_abstract = '\n'.join(map(str, release_notes['Release Note Abstract']))

            # Split changlog into sentences and remove duplicates
            changelog_sentences = list(set(changelog.split('\n')))
            changelog_sentences_abstract = list(set(changelog_abstract.split('\n')))
            
            # Remove linking statements or empty sentences
            changelog_sentences = [sentence for sentence in changelog_sentences  
                                   if not (any(re.match(pattern, sentence) for pattern in linking_statement) or sentence == '')]
            changelog_sentences_abstract = [sentence for sentence in changelog_sentences_abstract
                                            if not (any(re.match(pattern, sentence) for pattern in linking_statement) or sentence == '')]
            
            # Encode changelog sentences
            print('Start to encode changelog sentences')
            encoded_changelog_sentences = model.encode(changelog_sentences, convert_to_tensor=True)
            encoded_changelog_sentences_abstract = model.encode(changelog_sentences_abstract, convert_to_tensor=True)
            print('Successfully encoded changelog sentences')
            
            # Check encoded changelog sentences results
            print('Encoded changelog sentences shape:', encoded_changelog_sentences.shape)
            print('Encode changelog sentences abstract shape:', encoded_changelog_sentences_abstract.shape)

            # Load commit messages
            commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'commits.csv')
            commits = pd.read_csv(commits_path)
            normal_commits = commits[['Message', 'Description']]
            abstract_commits = commits[['Message Abstract', 'Description Abstract']]
            normal_commits['Message'] = normal_commits['Message'].apply(lambda message: str(message))
            abstract_commits['Message Abstract'] = abstract_commits['Message Abstract'].apply(lambda message: str(message))
            
            # Drop commit message with same content:
            normal_commits.drop_duplicates(subset='Message', keep='first', inplace=True)
            abstract_commits.drop_duplicates(subset='Message Abstract', keep='first', inplace=True)
            
            # Encode commit messages
            print('Start to encode commit messages')
            encoded_commit_messages = model.encode(normal_commits['Message'].to_numpy(), convert_to_tensor=True)
            encoded_commit_messages_abstract = model.encode(abstract_commits['Message Abstract'].to_numpy(), convert_to_tensor=True)
            print('Successfully encoded commit messages')
            
            # Check encoded commit messages result
            print('Encoded commit messages shape:',encoded_commit_messages.shape)
            print('Encoded commit messages abtract shape:', encoded_commit_messages_abstract.shape)

            # Score commit messages without abstraction
            normal_cosine_similar = cosine_similarity(encoded_commit_messages, encoded_changelog_sentences)
            normal_score = np.amax(normal_cosine_similar, axis=1)
            normal_index = np.argmax(normal_cosine_similar, axis=1)
            normal_cm = np.asarray(normal_commits['Message'])
            normal_ccs = np.array([changelog_sentences[x] for x in normal_index])
            normal_cd = np.asarray(normal_commits['Description'])
            normal_label = np.where(normal_score >= THRESHOLD, 1, 0)
            df = pd.DataFrame({'Commit Message': normal_cm, 'Score': normal_score, 
                               'Correspond Changelog Sentence': normal_ccs, 
                               'Commit Description': normal_cd, 'Label': normal_label})
            print('Label with normal approach:', df.head())

            # Write label result to the dataset
            labeled_commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'labeled_commits.csv')
            df.to_csv(labeled_commits_path, index=False)
            
            # Score commit messages with abstraction
            abstract_cosine_similar = cosine_similarity(encoded_commit_messages_abstract, encoded_changelog_sentences_abstract)
            abstract_score = np.amax(abstract_cosine_similar, axis=1)
            abstract_index = np.argmax(abstract_cosine_similar, axis=1)
            abstract_cm = np.asarray(abstract_commits['Message Abstract'])
            abstract_ccs = np.array([changelog_sentences_abstract[x] for x in abstract_index])
            abstract_cd = np.asarray(abstract_commits['Description Abstract'])
            abstract_label = np.where(abstract_score >= THRESHOLD, 1, 0)

            df = pd.DataFrame({'Commit Message': abstract_cm, 'Score': abstract_score, 
                               'Correspond Changelog Sentence': abstract_ccs, 
                               'Commit Description': abstract_cd, 'Label': abstract_label})
            print('Label with abstract approach:', df.head())
            # Write label result to the dataset
            labeled_commits_abstract_path = os.path.join(ROOT_DIR, 'data', folder_name, 'labeled_commits_abstract.csv')
            df.to_csv(labeled_commits_abstract_path, index=False)

            # Mark as has labeled commits
            corpus_repo_training.loc[i, 'Label status'] = 'Done'
            corpus_repo_training.to_csv(corpus_path, index=False)
            


        
