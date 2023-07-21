import pandas as pd
import os
import requests
import pygit2
from datetime import datetime
from settings import HEADERS, ROOT_DIR
from bs4 import BeautifulSoup
from markdown import markdown
from typing import List, Callable, Optional, Tuple, TypeVar

Time = TypeVar("Time")


def github_api(owner: str, repo: str, type: str, func: Callable, option: str = "") -> List[str]:
    """ Get all specific component of element has type is type using github_api """

    page = 1
    all_els = []
    while True:
        url ="https://api.github.com/repos/"\
             f"{owner}/{repo}/{type}?{option}&per_page=100&page={page}"
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
                       "body": el["body"], "created_at": el["created_at"]}
    
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
    remote_branches = [branch.split('/')[-1] for branch in all_branches[1:]]
    
    all_commit_shas = set()
    for branch in target_branches[:2]:
        if any([branch == rb for rb in remote_branches]):
            cmd = f"""cd {path} 
            git rev-list remotes/origin/{branch}"""
        else:
            "In here"
            cmd = f"""cd {path} 
            git rev-list {branch}"""
        commit_shas = os.popen(cmd).read()
        # Each line is a commit sha and the last line is empty line
        commit_shas = commit_shas.split('\n')[:-1]
        all_commit_shas.update(commit_shas)
         
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

    
    return commits


# def crawl_changelogs(owner, repo):
#     path = os.path.join(ROOT_DIR, "..", "data", f"{owner}_{repo}")
#     cmd = f"cd {path} \
#         git for-each-ref refs/tags/v2.2.0 --format='%(contents)'"
#     result = os.popen(cmd).read()
#     print(result)


def get_commit(commit: str) -> Optional[Tuple[str, str]]:
    """ Split commit into commit message (the first line) and follow by commit description """

    try:
        print("In here")
        # Convert markdown into html
        html = markdown(commit)
        soup = BeautifulSoup(html, "html.parser")
        lines = [p.text.strip() for p in soup.find_all('p')]
        message = lines[0]
        description = "<.> ".join(lines[1:])

        return message, description
    except:

        return None


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


def crawl_changelog_info() -> None:
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    # Check repos
    print(repos.shape)
    print(repos.head())
    num_repo = repos.shape[0]
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder = f"{owner}_{repo}"
        print("Repo:", owner, repo)
        try:
            # Crawl changelogs
            print("Start crawl changelogs")
            changelog_info = crawl_changelogs(owner, repo)
            print("Crawl changelogs done")
            changelog_info = pd.DataFrame(changelog_info)
            changelog_info.index.name = "Index"
            print(changelog_info.head())
            
            folder_path = os.path.join(ROOT_DIR, "data", folder)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            ci_path = os.path.join(folder_path, "changelog_info.csv")
            changelog_info.to_csv(ci_path)
            repos["Crawl status"] = "Done"
        except Exception as e:
            print(e)
            repos.loc[i, "Crawl status"] = "Error"


def crawl_data() -> None:   
    """ Crawl commits and changelog sentences of all repositories in "Repos.csv" file """
    
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    # Check repos
    print(repos.shape)
    print(repos.head())
    num_repo = repos.shape[0]
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
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
        
            # # Get changelog sentences
            # # cs_arr is sort for changelog sentences array
            # cs_arr = [get_changelog_sentences(changelog) for changelog in changelog_info.loc[:, "body"]]  
            # changelog_sentences = [sentence for cs in cs_arr for sentence in cs]
            # changelog_sen_df = pd.DataFrame({"Changelog Sentence": changelog_sentences})
            # changelog_sen_df = changelog_sen_df.drop_duplicates(subset=["Changelog Sentence"], keep="first")\
            #                              .reset_index(drop=True)
            # # Check changelog sentences
            # print("Num changelog sentences:", len(changelog_sen_df))
    
            # Crawl commits
            print("Start load commits")
            commits = crawl_commits(owner, repo, target_branches, lastest, oldest)
            print("Commits loaded")
            print(len(commits))
            # Get commit messages and commit descriptions
            commits = [get_commit(commit["message"]) 
                       for commit in commits if get_commit(commit["message"])]
            print(len(commits))
            messages, descriptions = zip(*commits)
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
            repos.loc[i, "Crawl status"] = "Done"
        except Exception as e:
            print(e)
            repos.loc[i, "Crawl status"] = "Error"
        break        
        repos.to_csv(repos_path, index=False)


crawl_data()