import os
import requests
import pygit2
import pandas as pd
import numpy as np
from settings import HEADERS, ROOT_DIR, MODEL, LABEL_THRESHOLD
from bs4 import BeautifulSoup
from markdown import markdown
from sklearn.metrics.pairwise import cosine_similarity
from base_functions import traverse_repos
from typing import List, Callable,  Tuple, TypeVar


Time = TypeVar("Time")

class MyRemoteCallbacks(pygit2.RemoteCallbacks):
    def transfer_progress(self, stats):
        print(f'{stats.indexed_objects}/{stats.total_objects}')


def github_api(owner: str, repo: str, type: str, func: Callable, option: str = "") -> List[str]:
    """ Get all specific component of element has type is type using github_api """

    page = 1
    all_els = []
    while True:
        url ="https://api.github.com/repos/{owner}/{repo}/{type}?{option}&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        els = response.json()
        els_per_page = [func(el) for el in els]
        all_els += els_per_page
        # 100 is the limit of per_page param in github api
        if len(els) < 100:  
            break
        page += 1

    return all_els


def crawl_changelogs(owner: str, repo: str) -> Callable[[str, str, str, Callable], List[str]]:
    """ Crawl all changelogs in https://github.com/owner/repo"""
    
    func = lambda el: {"tag_name": el["tag_name"], "target_commitish": el["target_commitish"],
                       "body": el["body"], "created_at": el["created_at"], "published_at": el["published_at"]}
    
    return github_api(owner, repo, type="releases", func=func)


# def crawl_branches(owner, repo):
#     return github_api(owner, repo, type="branches", func=lambda el: el["name"])


# def crawl_commits(owner, repo):
#     branches = crawl_branches(owner, repo)
#     print("Num branches:", branches)
#     all_commits = set()
#     for branch in branches:
#         commits = github_api(owner, repo, type="commits", 
#                              func=lambda el: el["commit"]["message"], 
#                              option=f"sha={branch}")
#         print(f"Number commits in branch {branch}:", len(commits))
#         all_commits.update(commits)
#     return all_commits


def crawl_commits(owner: str, repo: str, target_branches: List[str], lastest: Time, oldest: Time) -> List[str]:
    """ Crawl all commits in https://github.com/owner/repo """

    path = os.path.join(ROOT_DIR, "..", "repos", f"{owner}_{repo}")
    cmd = f""" cd {path} 
            git branch -a"""
    all_branches = os.popen(cmd).read().split('\n')[:-1]
    all_branches = [branch.strip() for branch in all_branches]
    remote_branches = ['/'.join(branch.split('/')[2:]) for branch in all_branches[1:]]
    
    all_commit_shas = set()
    for branch in target_branches:
        try:
            if any(rb == branch for rb in remote_branches):
                cmd = f"""cd {path} 
                git rev-list remotes/origin/{branch}"""
            else:
                cmd = f"""cd {path} 
                git rev-list {branch}"""
            commit_shas = os.popen(cmd).read()
            # Each line is a commit sha and the last line is empty line
            commit_shas = commit_shas.split('\n')[:-1]
            all_commit_shas.update(commit_shas)
        except:
            continue
         
    repo = pygit2.Repository(path)
    # Get commit message from commit sha
    commits = [repo.revparse_single(commit_sha) for commit_sha in all_commit_shas]

    def valid(commit, lastest, oldest):
        return oldest.to_pydatetime().timestamp() <= commit.commit_time <= lastest.to_pydatetime().timestamp()

    # Get all commit message and commit sha
    commits = [
        {
            "message": commit.message, 
            "sha": commit.hex, 
            "author": commit.author, 
            "commit_time": commit.commit_time, 
            "committer": commit.committer
        } 
        for commit in commits if valid(commit, lastest, oldest)
    ]
    commits = pd.DataFrame(commits)

    return commits


def get_commit(commit: str) -> Tuple[str, str]:
    """ Split commit into commit message (the first line) and follow by commit description """

    try:
        # Convert markdown into html
        html = markdown(commit)
        soup = BeautifulSoup(html, "html.parser")
        lines = [p.text.strip() for p in soup.find_all('p')]
        message = lines[0]
        description = "<.> ".join(lines[1:])

        return message, description
    except:

        return None, None


def get_changelog_sentences(changelog: str) -> List[str]:
    """Get changelog sentences from raw changelog """

    try:
        html = markdown(changelog)
        soup = BeautifulSoup(html, "html.parser")
        sentences_li = [li.text.strip() for li in soup.find_all("li")]
        sentences_p = [p.text.strip().split("\n") for p in soup.find_all('p')]
        sentences_p = [sentence[i] for sentence in sentences_p for i in range(len(sentence))]
        sentences = [*sentences_li, *sentences_p]

        return sentences
    except:

        return []


def make_data() -> None:
    """ Functions to make data """

    def clone_repos(owner: str, repo: str) -> None:
        path = os.path.join(ROOT_DIR, "..", "repos", f"{owner}_{repo}")
        pygit2.clone_repository(f"https://github.com/{owner}/{repo}", path, callbacks=MyRemoteCallbacks())
    
    def crawl_changelog_info(owner: str, repo: str) -> None:
        """ Get information of changelogs in https://github.com/owner/repo"""

        folder = f"{owner}_{repo}"
        print("Repo:", owner, repo)
        try:
            # Crawl changelogs
            print("Start crawl changelogs")
            changelog_info = crawl_changelogs(owner, repo)
            print("Crawl changelogs done")
            changelog_info = pd.DataFrame(changelog_info)
            changelog_info.index.name = "index"
            print(changelog_info.head())
            
            folder_path = os.path.join(ROOT_DIR, "data", folder)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            ci_path = os.path.join(folder_path, "changelog_info.csv")
            changelog_info.to_csv(ci_path)
        except Exception as e:
            raise e

    def crawl_data(owner: str, repo: str) -> None:   
        """ Crawl commits and changelog sentences of all repositories in "Repos.csv" file """
        
        folder = f"{owner}_{repo}"
        print("Repo:", owner, repo)
        try:
            # Load changelogs
            print("Start load changelogs")
            folder_path = os.path.join(ROOT_DIR, "data", folder)
            changelog_info_path = os.path.join(folder_path, "changelog_info.csv")
            changelog_info = pd.read_csv(changelog_info_path)
            print("Changelogs loaded")
            changelog_info["created_at"] = pd.to_datetime(changelog_info["created_at"])

            lastest = changelog_info["created_at"].max()
            oldest = changelog_info["created_at"].min()
            target_branches = changelog_info["target_commitish"].unique().tolist()
        
            # Get changelog sentences
            # cs_arr is sort for changelog sentences array
            cs_arr = [get_changelog_sentences(changelog) for changelog in changelog_info.loc[:, "body"]]  
            changelog_sentences = [sentence for cs in cs_arr for sentence in cs]
            changelog_sen_df = pd.DataFrame({"Changelog Sentence": changelog_sentences})
            changelog_sen_df = changelog_sen_df.drop_duplicates(subset=["Changelog Sentence"], keep="first")\
                                        .reset_index(drop=True)
            # Check changelog sentences
            print("Num changelog sentences:", len(changelog_sen_df))

            # Crawl commits
            print("Start load commits")
            commits = crawl_commits(owner, repo, target_branches, lastest, oldest)
            print("Commits loaded")
            # Get commit messages and commit descriptions
            mes_des = [get_commit(commit) 
                    for commit in commits.loc[:, "message"]]
            messages, descriptions = zip(*mes_des)

            commit_df = pd.DataFrame({
                "Owner": owner,
                "Repo": repo,
                "Message": messages, 
                "Description": descriptions,
                "Sha": commits["sha"],
                "Author": commits["author"],
                "Committer": commits["committer"],
                "Commit Time": commits["commit_time"]
            })
            commit_df = commit_df.dropna(subset=["Message"]).reset_index(drop=True)
            commit_df = commit_df.drop_duplicates(subset=["Message"]).reset_index(drop=True)
            # Check commit messages
            print("Num commit messages:", len(commit_df))
            print("\n")
            print("==============================================")
            print("\n")

            # Save data to folder
            # changelog_sen_path = os.path.join(folder_path, "changelog_sentence.csv")
            # changelog_sen_df.index.name = "Index"
            # changelog_sen_df.to_csv(changelog_sen_path)
            commit_path = os.path.join(folder_path, "commit.csv")
            commit_df.index.name = "Index"
            commit_df.to_csv(commit_path)
        except Exception as e:
            raise e

    def label_commit(owner: str, repo: str) -> None:
        """ Label commit using pretrained sentence-BERT model and cosine similarity function """

        folder = f"{owner}_{repo}"
        print("Repo:", owner, repo)
        try:
            # Load and encode changelog sentences
            c_log_path = os.path.join(ROOT_DIR, "data", folder, "changelog_sentence.csv")
            c_log_sen = pd.read_csv(c_log_path)["Changelog Sentence"].astype("str")
            # Encode changelog sentences
            print("Start to encode changelog sentences")
            encoded_c_log_sen = MODEL.encode(c_log_sen, convert_to_numpy=True)
            print("Successfully encoded changelog sentences")
            # Check encoded changelog sentences results
            print("Encoded changelog sentences shape:", encoded_c_log_sen.shape)
            
            # Load commit messages
            commits_path = os.path.join(ROOT_DIR, "data", folder, "commit.csv")
            commits = pd.read_csv(commits_path).astype("str")
            # Encode commit messages
            print("Start to encode commit messages")
            encoded_cm = MODEL.encode(commits["Message"], convert_to_numpy=True)
            print("Successfully encoded commit messages")
            # Check encoded commit messages result
            print("Encoded commit messages shape:", encoded_cm.shape)

            # Calculate cosine similarity
            scores = cosine_similarity(encoded_cm, encoded_c_log_sen)
            # Score is the max of cosine similarity scores between encoded commit and encoded changelog sentences 
            max_scores = np.amax(scores, axis=1) 
            # Index of corresponding changelog sentence that result max cosine similarity score with commits
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
            labelled_commits_path = os.path.join(ROOT_DIR, "data", folder, "labelled_commit.csv")
            df.to_csv(labelled_commits_path, index=False)
        except Exception as e:
            raise e
    
    traverse_repos(label_commit)


# def join_labelled_data():
#     """ Join labelled commits of all repositories into one file """

#     # Load all repositories commits into dataframes
#     repo_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
#     repos = pd.read_csv(repo_path)
#     num_repo = len(repos)
#     dfs = []
#     for i in range(num_repo):
#         owner = repos.loc[i, "Owner"]
#         repo = repos.loc[i, "Repo"]
#         path = os.path.join(ROOT_DIR, "data", f"{owner}_{repo}", "labelled_commit.csv")
#         df = pd.read_csv(path)[["Commit Message", "Description", "Score", "Label"]]
#         dfs.append(df)

#     # Merge all repositories commits into one dataframe 
#     all_data = pd.concat(dfs)
#     # Remove null commit has no message
#     all_data = all_data.dropna(subset=["Commit Message"]).reset_index(drop=True)
#     # Remove duplicate commits has same message
#     all_data = all_data.drop_duplicates(subset=["Commit Message"]).reset_index(drop=True)
#     # Shuffle commits
#     all_data = all_data.sample(frac=1, axis=0).reset_index(drop=True)
#     # Number commits
#     print(all_data.head())
#     all_data.to_csv(os.path.join(ROOT_DIR, "data", "dataset.csv"), index=False)


# def join_labelled_data_by_repo():
#     """ Join labelled commits of all repositories into train.csv and test.csv by repo """

#     # Load all repositories commits into dataframes
#     repo_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
#     repos = pd.read_csv(repo_path)
#     num_repo = len(repos)
#     train_dfs = []
#     test_dfs = []
#     repos = repos.sample(frac=1, axis=0, random_state=1)
#     repos = repos.set_index(np.arange(len(repos.index))) 
#     num_train_repo = int(num_repo * 0.8)
#     print(num_train_repo)
#     # for i in range(num_repo):
#     #     owner = repos.loc[i, "Owner"]
#     #     repo = repos.loc[i, "Repo"]
#     #     path = os.path.join(ROOT_DIR, "data", f"{owner}_{repo}", "labelled_commit.csv")
#     #     df = pd.read_csv(path)[["Commit Message", "Description", "Score", "Label"]]
#     #     if i < num_train_repo:
#     #         train_dfs.append(df)
#     #     else:
#     #         test_dfs.append(df)
#     # all_data = pd.concat(train_dfs)
#     # # Remove null commit has no message
#     # all_data = all_data.dropna(subset=["Commit Message"]).reset_index(drop=True)
#     # # Remove duplicate commits has same message
#     # all_data = all_data.drop_duplicates(subset=["Commit Message"]).reset_index(drop=True)
#     # # Shuffle commits
#     # all_data = all_data.sample(frac=1, axis=0).reset_index(drop=True)
#     # # Number commits
#     # print(all_data.head())
#     # all_data.to_csv(os.path.join(ROOT_DIR, "data", "train_repo_dataset.csv"), index=False)

#     # test_data = pd.concat(test_dfs)
#     # # Remove null commit has no message
#     # test_data = test_data.dropna(subset=["Commit Message"]).reset_index(drop=True)
#     # # Remove duplicate commits has same message
#     # test_data = test_data.drop_duplicates(subset=["Commit Message"]).reset_index(drop=True)
#     # # Shuffle commits
#     # test_data = test_data.sample(frac=1, axis=0).reset_index(drop=True)
#     # # Number commits
#     # print(test_data.head())
#     # test_data.to_csv(os.path.join(ROOT_DIR, "data", "test_repo_dataset.csv"), index=False)
                

def join_labelled_data_by_time():
    """ Join labelled commits of all repositories into train.csv and test.csv by repo """

    # Load all repositories commits into dataframes
    repo_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repo_path)
    num_repo = len(repos)
    dfs = []
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        print("Repo:", owner, repo)
        path = os.path.join(ROOT_DIR, "data", f"{owner}_{repo}", "labelled_commit.csv")
        df = pd.read_csv(path)
        commit_time = []
        path = os.path.join(ROOT_DIR, "..", "repos", f"{owner}_{repo}")
        for j in range(len(df)):
            cmd = f"""cd {path}
                git show --no-patch --no-notes --format="%ct" {df.loc[j, "Sha"]}"""
            time = int(os.popen(cmd).read()[:-1])
            commit_time.append(time)
        df["Commit Time"] = pd.to_datetime(commit_time, unit='s')
        dfs.append(df)
        
    all_data = pd.concat(dfs)
    # Remove null commit has no message
    all_data = all_data.dropna(subset=["Commit Message"]).reset_index(drop=True)
    # Remove duplicate commits has same message
    all_data = all_data.drop_duplicates(subset=["Commit Message"]).reset_index(drop=True)
    # Shuffle commits
    all_data = all_data.sample(frac=1, axis=0).reset_index(drop=True)
    # Number commits
    all_data = all_data.sort_values(by=["Commit Time"], ascending=True).reset_index(drop=True)
    print(all_data.head())
    all_data.to_csv(os.path.join(ROOT_DIR, "data", "dataset_by_time.csv"), index=False)

