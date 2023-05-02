import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


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
            changelog_sentences = list(set(changelog.split('\n')))
            print('Start to encode changelog sentences')
            encoded_changelog_sentences = model.encode(changelog_sentences)
            print('Successfully encoded changelog sentences')
            print('Encoded changelog sentences shape:', encoded_changelog_sentences.shape)

            # Load and encode commit messages
            commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'commits.csv')
            commits = pd.read_csv(commits_path)
            commit_messages = commits['Message'].apply(lambda message: str(message))

            # New: Add filter to drop commit message with same content:
            commit_messages = list(set(commit_messages)) 
            commit_description = commits['Description']
            print('Start to encode commit messages')
            encoded_commit_messages = model.encode(commit_messages)
            print('Successfully encoded commit messages')
            print('Encoded commit messages shape:',encoded_commit_messages.shape)
            
            # Score commit messages
          
            results = []
            for i in range(len(encoded_commit_messages)):
                result = list(cosine_similarity([encoded_commit_messages[i]], encoded_changelog_sentences)[0])
                score = max(result)
                index_max = result.index(score)
                cm = commit_messages[i]
                ccs = changelog_sentences[index_max]
                cd = commit_description[i]
                new_record = [i + 1, cm, score, ccs, cd]
                results.append(new_record)

            df = pd.DataFrame(results, 
                              columns=['Index', 'Commit Message', 'Score', 'Correspond Changelog Sentence', 'Commit Description'])
            print('Results:', df.head())
            labeled_commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'labeled_commits.csv')
            df.to_csv(labeled_commits_path, index=False)    

            print('Current index:', index)
            corpus_repo_training.loc[index, 'Label status'] = 'Done'
            print('Updated corpus_repo_training shape:', corpus_repo_training.shape)
            corpus_repo_training.to_csv(corpus_path, index=False)
            
        index = (index + 1) % num_repo
        
