import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from settings import ROOT_DIR, MODEL, LABEL_THRESHOLD


def label_commit() -> None:
    """ Label commit using pretrained sentence-BERT model and cosine similarity function """

    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    num_repo = repos.shape[0]
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder = f"{owner}_{repo}"
        print("Repo:", owner, repo)
        try:
            # Load and encode changelog sentences
            c_log_path = os.path.join(ROOT_DIR, "data", folder, "changelogs.csv")
            c_log_sen = pd.read_csv(c_log_path)["Changelog Sentence"].astype("str")
            # Encode changelog sentences
            print("Start to encode changelog sentences")
            encoded_c_log_sen = MODEL.encode(c_log_sen, convert_to_numpy=True)
            print("Successfully encoded changelog sentences")
            # Check encoded changelog sentences results
            print("Encoded changelog sentences shape:", encoded_c_log_sen.shape)
            
            # Load commit messages
            commits_path = os.path.join(ROOT_DIR, "data", folder, "commits.csv")
            commits = pd.read_csv(commits_path).astype("str")
            # Encode commit messages
            print("Start to encode commit messages")
            encoded_cm = MODEL.encode(commits["Message"], convert_to_numpy=True)
            print("Successfully encoded commit messages")
            # Check encoded commit messages result
            print("Encoded commit messages shape:", encoded_cm.shape)

            # Calculate cosine similarity
            scores = cosine_similarity(encoded_cm, encoded_c_log_sen)
            # Score is the max of cosine similarity scores 
            # between encoded commit and encoded changelog sentences 
            max_scores = np.amax(scores, axis=1) 
            # Index of corresponding changelog sentence 
            # that result max cosine similarity score with commits
            index = np.argmax(scores, axis=1) 
            cm = np.asarray(commits["Message"])  # Commit messages
            # Corresponding changelog sentences
            ccs = np.array([c_log_sen[x] for x in index])  
            # Commit descriptions
            addition_info = commits[["Description", "Owner", "Repo", "Sha"]]
            # Label of commits 
            label = np.where(max_scores >= LABEL_THRESHOLD, 1, 0) 
            idx = np.asarray(range(1, len(cm) + 1))  # Index
            df = pd.DataFrame({"Index": idx,
                               "Commit Message": cm, 
                               "Score": max_scores, 
                               "Correspond Changelog Sentence": ccs, 
                               "Label": label
                            })
            df = pd.concat([df, addition_info], axis=1)
            print("Labelled commit:", df.head())

            # Write label result to the dataset
            labelled_commits_path = os.path.join(ROOT_DIR, "data", folder, "labelled_commits.csv")
            df.to_csv(labelled_commits_path, index=False)
            repos.loc[i, "Label status"] = "Done"
        except Exception as e:
            repos.loc[i, "Label status"] = "Error"
    
        repos.to_csv(repos_path, index=False)

    
def score_similarity() -> None:
    """ Calculate scores of sample commits """

    sample_path = os.path.join(ROOT_DIR, "data", "sample_data.csv")
    sample_dataset = pd.read_csv(sample_path).astype("str")
    sample_commit_path = os.path.join(ROOT_DIR, "data", "sample_commit")
    if not os.path.exists(sample_commit_path):
        os.mkdir(sample_commit_path)
    for i in range(len(sample_dataset)):
        _, sha, owner, repo, message, description = sample_dataset.loc[i]
        
        # Load changelog sentences
        folder = f"{owner}_{repo}"
        print(folder)
        c_log_path = os.path.join(ROOT_DIR, "data", folder, "changelogs.csv")
        c_log_sen = pd.read_csv(c_log_path)["Changelog Sentence"].to_numpy().astype("str")

        # Score commit for each changelog sentences
        encoded_mes = MODEL.encode([message], convert_to_numpy=True)
        encoded_c_log_sen = MODEL.encode(c_log_sen, convert_to_numpy=True)
        scores = cosine_similarity(encoded_mes, encoded_c_log_sen)[0]
        scores = np.concatenate((c_log_sen.reshape(-1, 1), scores.reshape(-1, 1)), axis=1)
        c_log_sen = pd.DataFrame(scores, columns=["Attribute", "Value"])
        c_log_sen["Value"] = pd.to_numeric(c_log_sen["Value"])
        c_log_sen = c_log_sen.sort_values(by="Value", ascending=False)
        header = {
            "Label": "",
            "Message": f"{message}",
            "Description": f"{description}",
            "Sha": f"{sha}",
            "Owner": f"{owner}",
            "Repo": f"{repo}",
            "": "",
            "Changelog Sentence": "Score"
        }
        header = np.array(list(header.items()))
        header = pd.DataFrame(header, columns=["Attribute", "Value"])
        record = pd.concat([header, c_log_sen], axis=0)
        record["Index"] = [idx + 1 for idx in record.index]
        record = record[["Index", "Attribute", "Value"]]
        print(record.shape)
        record.to_csv(os.path.join(sample_commit_path, f"test_{i + 1}.csv"), index=False)

score_similarity()
