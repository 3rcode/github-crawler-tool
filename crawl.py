import pandas as pd
import os
import requests
from collections import defaultdict
import pygit2
from settings import HEADERS, ROOT_DIR
from bs4 import BeautifulSoup
from markdown import markdown
from typing import List, Callable, Optional, Tuple


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

    return github_api(owner, repo, type="releases", func=lambda el: el["body"])


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


def crawl_commits(owner: str, repo: str) -> List[str]:
    """ Crawl all commits in https://github.com/owner/repo """

    path = os.path.join(ROOT_DIR, "..", "data", f"{owner}_{repo}")
    cmd = f"""cd {path} 
        git rev-list --branches=* --remotes=*"""
    commit_shas = os.popen(cmd).read()
    # Each line is a commit sha and the last line is empty line
    commit_shas = commit_shas.split('\n')[:-1]  
    repo = pygit2.Repository(path)
    # Get commit message from commit sha
    commits = [repo.revparse_single(commit_sha) for commit_sha in commit_shas]
    # Get all commit message and commit sha
    commits = [{"message": commit.message, "sha": commit.hex} for commit in commits]
    
    return commits


# def crawl_changelogs(owner, repo):
#     path = os.path.join(ROOT_DIR, "..", "data", f"{owner}_{repo}")
#     cmd = f"cd {path} \
#         git for-each-ref refs/tags/v2.2.0 --format='%(contents)'"
#     result = os.popen(cmd).read()
#     print(result)


def get_commit(commit: str, sha: str) -> Optional[Tuple[str, str, str]]:
    """ Split commit into commit message (the first line) and follow by commti description """

    try:
        # Convert markdown into html
        html = markdown(commit)
        soup = BeautifulSoup(html, "html.parser")
        lines = [p.text.strip() for p in soup.find_all('p')]
        message = lines[0]
        description = "<.> ".join(lines[1:])

        return sha, message, description
    except:

        return None


def get_changelog_sentences(changelog: str) -> List[str]:
    """Get changelog sentences from raw changelog """

    try:
        html = markdown(changelog)
        soup = BeautifulSoup(html, "html.parser")
        sentences_li = [li.text.strip() 
                        for li in soup.find_all("li")]
        sentences_p = [p.text.strip().split("\n") 
                    for p in soup.find_all('p')]
        sentences_p = [sentence[i] 
                    for sentence in sentences_p 
                        for i in range(len(sentence))]
        sentences = [*sentences_li, *sentences_p]

        return sentences
    except:

        return []


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
            # Crawl changelogs
            print("Start crawl changelogs")
            changelogs = crawl_changelogs(owner, repo)
            print("Crawl changelogs done")
            # Get changelog sentences
            # cs_arr is sort for changelog sentences array
            cs_arr = [get_changelog_sentences(changelog) for changelog in changelogs]  
            changelog_sentences = [sentence for cs in cs_arr for sentence in cs]
            changelogs_df = pd.DataFrame({"Changelog Sentence": changelog_sentences})
            changelogs_df = changelogs_df.drop_duplicates(subset=["Changelog Sentence"], keep="first")\
                                         .reset_index(drop=True)
            changelogs_df["Index"] = [idx + 1 for idx in changelogs_df.index]
            changelogs_df = changelogs_df[["Index", "Changelog Sentence"]]
            # Check changelog sentences
            print("Num changelog sentences:", len(changelogs_df))

            # Crawl commits
            print("Start crawl commits")
            commits = crawl_commits(owner, repo)
            print("Crawl commits done")
            # Get commit messages and commit descriptions
            commits = [get_commit(commit["message"], commit["sha"]) 
                       for commit in commits if get_commit(commit["message"], commit["sha"])]
            shas, messages, descriptions = zip(*commits)
            commits_df = pd.DataFrame({
                "Sha": shas,
                "Owner": owner,
                "Repo": repo,
                "Message": messages, 
                "Description": descriptions
        
            })
            commits_df = commits_df.drop_duplicates(subset=["Message"]).reset_index(drop=True)
            commits_df["Index"] = [idx + 1 for idx in commits_df.index]
            commits_df = commits_df[["Index", "Sha", "Owner", "Repo", "Message", "Description"]]
            # Check commit messages
            print("Num commit messages:", len(commits_df))
            print("\n")
            print("==============================================")
            print("\n")

            # Save data to folder
            folder_path = os.path.join(ROOT_DIR, "data", folder)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            c_logs_path = os.path.join(folder_path, "changelogs.csv")
            commits_path = os.path.join(folder_path, "commits.csv")
            changelogs_df.to_csv(c_logs_path, index=False)
            # commits_df.to_csv(commits_path, index=False)
            repos.loc[i, "Crawl status"] = "Done"
        except Exception as e:
            print(e)
            repos.loc[i, "Crawl status"] = "Error"
        
        repos.to_csv(repos_path, index=False)



    