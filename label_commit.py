import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
linking_statement = [r"(?i)fixes:?", r"(?i)what's changed:?", r"(?i)other changes:?", r"(?i)documentation:?",
                     r"(?i)features:?", r"(?i)new contributors:?", r"(?i)bug fixes:?", r"(?i)changelog:?",
                     r"(?i)notes:?", r"(?i)improvements:?", r"(?i)deprecations:?", r"(?i)minor updates:?",
                     r"(?i)changes:?", r"(?i)fixed:?", r"(?i)maintenance:?", r"(?i)added:?"]

THRESHOLD = 0.7

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(ROOT_DIR, 'data', 'Corpus Repo - Training.csv')
    corpus_repo_training = pd.read_csv(corpus_path)
    # Check corpus repo training
    # print(corpus_repo_training.shape)
    # print(corpus_repo_training.head())
    
    num_repo = corpus_repo_training.shape[0]
    index = 0
    
    while any(label_status != 'Done' for label_status in corpus_repo_training['Label status']):
        if (corpus_repo_training.loc[index, 'Crawl status'] == 'Done' 
            and corpus_repo_training.loc[index, 'Label status'] != 'Done'):
            owner = corpus_repo_training.loc[index, 'User']
            repo = corpus_repo_training.loc[index, 'Repo name']
            folder_name = f'{owner}_{repo}'
            print(owner, repo)

            # Load and encode changelog sentences
            release_notes_path = os.path.join(ROOT_DIR, 'data', folder_name, 'release_notes.csv')
            release_notes = pd.read_csv(release_notes_path)
            changelog = '\n'.join(map(str, release_notes['Release Note']))
            changelog_abstract = '\n'.join(map(str, release_notes['Release Note With Abstraction']))

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
            encoded_changelog_sentences = model.encode(changelog_sentences)
            encoded_changelog_sentences_abstract = model.encode(changelog_sentences_abstract)
            print('Successfully encoded changelog sentences')
            
            # Check encoded changelog sentences results
            print('Encoded changelog sentences shape:', encoded_changelog_sentences.shape)
            print('Encode changelog sentences abstract shape:', encoded_changelog_sentences_abstract.shape)

            # Load commit messages
            commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'commits.csv')
            commits = pd.read_csv(commits_path)
            commit_messages = commits['Message'].apply(lambda message: str(message))
            commit_messages_abstract = commits['Message Abstract'].apply(lambda message: str(message))
            
            # Drop commit message with same content:
            commit_messages = list(set(commit_messages)) 
            commit_messages_abstract = list(set(commit_messages_abstract))
            
            # Load corresponding commit description for commit message
            commit_description = commits['Description']
            commit_description_abstract = commits['Description Abstract']
            
            # Encode commit messages
            print('Start to encode commit messages')
            encoded_commit_messages = model.encode(commit_messages)
            encoded_commit_messages_abstract = model.encode(commit_messages_abstract)
            print('Successfully encoded commit messages')
            
            # Check encoded commit messages result
            print('Encoded commit messages shape:',encoded_commit_messages.shape)
            print('Encoded commit messages abtract shape:', encoded_commit_messages_abstract.shape)

            # Score commit messages without abstraction
            results = []
            for i in range(len(encoded_commit_messages)):
                result = list(cosine_similarity([encoded_commit_messages[i]], encoded_changelog_sentences)[0])
                score = max(result)
                index_max = result.index(score)
                cm = commit_messages[i]
                ccs = changelog_sentences[index_max]
                cd = commit_description[i]
                label = 1 if score > THRESHOLD else 0
                new_record = [i + 1, cm, score, ccs, cd, label]
                results.append(new_record)

            df = pd.DataFrame(results, 
                              columns=['Index', 'Commit Message', 'Score', 'Correspond Changelog Sentence', 'Commit Description', 'Label'])
            print('Results:', df.head())

            # Write label result to the dataset
            labeled_commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'labeled_commits.csv')
            df.to_csv(labeled_commits_path, index=False)
            
            results_abstract = []
            # Score commit messages with abstraction
            for i in range(len(encoded_commit_messages_abstract)):
                result = list(cosine_similarity([encoded_commit_messages_abstract[i]], encoded_changelog_sentences_abstract)[0])
                score = max(result)
                index_max = result.index(score)
                cm = commit_messages_abstract[i]
                ccs = changelog_sentences_abstract[index_max]
                cd = commit_description_abstract[i]
                label = 1 if score > THRESHOLD else 0
                new_record = [i + 1, cm, score, ccs, cd, label]
                results_abstract.append(new_record)

            df = pd.DataFrame(results_abstract, 
                              columns=['Index', 'Commit Message Abstract', 'Score Abstract', 
                                       'Corresponding Changelog Sentence Abstract', 'Commit Desription Abstract', 'Label'])
            print('Results:', df.head())
            # Write label result to the dataset
            labeled_commits_abstract_path = os.path.join(ROOT_DIR, 'data', folder_name, 'labeled_commits_abstract.csv')
            df.to_csv(labeled_commits_abstract_path, index=False)

            # Mark as has labeled commits
            corpus_repo_training.loc[index, 'Label status'] = 'Done'
            corpus_repo_training.to_csv(corpus_path, index=False)
            
        index = (index + 1) % num_repo


        
