import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from settings import ROOT_DIR, MODEL, LABEL_THRESHOLD


def label_commit():
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    num_repo = repos.shape[0]

    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder_name = f"{owner}_{repo}"
        print("Repo:", owner, repo)
        
        try:
            print(folder_name)
            # Load and encode changelog sentences
            changelog_path = os.path.join(ROOT_DIR, "data", folder_name, "changelogs.csv")
            changelog_sentences = pd.read_csv(changelog_path)["Changelog Sentence"]
            changelog_sentences = changelog_sentences.astype("str")
            
            # Encode changelog sentences
            print("Start to encode changelog sentences")
            encoded_changelog_sentences = MODEL.encode(changelog_sentences, convert_to_tensor=True)
            print("Successfully encoded changelog sentences")
            
            
            # Check encoded changelog sentences results
            print("Encoded changelog sentences shape:", encoded_changelog_sentences.shape)

            # Load commit messages
            commits_path = os.path.join(ROOT_DIR, "data", folder_name, "commits.csv")
            commits = pd.read_csv(commits_path)[["Message", "Description"]]
            # print(commits.info())
            commits = commits.astype("str")

            # Encode commit messages
            print("Start to encode commit messages")
            encoded_commit_messages = MODEL.encode(commits["Message"], convert_to_tensor=True)
            print("Successfully encoded commit messages")
            
            # Check encoded commit messages result
            print("Encoded commit messages shape:", encoded_commit_messages.shape)

            # Score commit messages without abstraction
            cosine_scores = cosine_similarity(encoded_commit_messages, encoded_changelog_sentences)

            # Score is the max of cosine similarity scores 
            # between encoded commit and encoded changelog sentences 
            score = np.amax(cosine_scores, axis=1)  

            # Index of corresponding changelog sentence 
            # that result max cosine similarity score with commits
            index = np.argmax(cosine_scores, axis=1) 
            cm = np.asarray(commits["Message"])  # Commit messages
            ccs = np.array([changelog_sentences[x] for x in index])  # Corresponding changelog sentences
            cd = np.asarray(commits["Description"])  # Commit descriptions
            label = np.where(score >= LABEL_THRESHOLD, 1, 0) # Label of commits
            df = pd.DataFrame({"Commit Message": cm, 
                            "Score": score, 
                            "Correspond Changelog Sentence": ccs, 
                            "Commit Description": cd, 
                            "Label": label
                            })
            print("Label commit:", df.head())

            # Write label result to the dataset
            labelled_commits_path = os.path.join(ROOT_DIR, "data", folder_name, "labelled_commits.csv")
            df.to_csv(labelled_commits_path, index=False)
            repos.loc[i, "Label status"] = "Done"
        except Exception as e:
            repos.loc[i, "Label status"] = "Error"
    
        repos.to_csv(repos_path, index=False)
        


        
