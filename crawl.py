import requests
import pandas as pd
import os
from settings import HEADERS, ROOT_DIR
from bs4 import BeautifulSoup
from markdown import markdown

def github_api(owner, repo, type, func, option=""):
    page = 1
    all_els = []
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/{type}?{option}&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        els = response.json()
        els_per_page = [func(el) for el in els]
        all_els += els_per_page
        if len(els) < 100:
            break
        page += 1
    return all_els

def crawl_changelogs(owner, repo):
    return github_api(owner, repo, type="releases", func=lambda el: el["body"])

def crawl_branches(owner, repo):
    return github_api(owner, repo, type="branches", func=lambda el: el["name"])

def crawl_commits(owner, repo):
    branches = crawl_branches(owner, repo)
    all_commits = set()
    for branch in branches:
        commits = github_api(owner, repo, type="commits", 
                             func=lambda el: el["commit"]["message"], 
                             option=f"sha={branch}")
        print(f"Number commits in branch {branch}:", len(commits))
        all_commits.update(commits)
    return all_commits

def get_commit(commit, replace=False):
    if not commit:
        print("Has none commit")
        return None
    html = markdown(commit)
    soup = BeautifulSoup(html, "html.parser")

    sentences = [p.text.strip() for p in soup.find_all('p')]
    if not sentences:
        print("Can't split commit by p tag")
        return None
    message = sentences[0]
    description = sentences[1:]
    description = "<.> ".join(description)
    return message, description

def get_changelog_sentences(changelog, replace=False):
    if not changelog:
        return []
    html = markdown(changelog)
    soup = BeautifulSoup(html, "html.parser")
    sentences_li = [li.text.strip() for li in soup.find_all("li")]
    sentences_p = [p.text.strip().split("\n") for p in soup.find_all('p')]
    sentences_p = [sentence[i] for sentence in sentences_p for i in range(len(sentence))]
    sentences = [*sentences_li, *sentences_p]
    return sentences

def crawl_data():
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    
    # Check repos
    print(repos.shape)
    print(repos.head())
    
    num_repo = repos.shape[0]

    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder_name = f"{owner}_{repo}"
        print("Repo:", owner, repo)
        try:
            # Crawl changelogs
            print("Start crawl changelogs")
            changelogs = crawl_changelogs(owner, repo)
            print("Crawl changelogs done")

            # Get changelog sentences
            cs_arr = [get_changelog_sentences(changelog) for changelog in changelogs]  # cs_arr is sort for changelog sentences array
            changelog_sentences = [sentence for cs in cs_arr for sentence in cs]
            idx = range(1, len(changelog_sentences) + 1)
            changelogs_df = pd.DataFrame({"Index": idx, "Changelog Sentence": changelog_sentences})
            changelogs_df = changelogs_df.drop_duplicates(subset=["Changelog Sentence"], keep="first").reset_index(drop=True)

            # Check changelog sentences
            print("Num changelog sentences:", len(changelogs_df))

            # Crawl commits
            print("Start crawl commits")
            commits = crawl_commits(owner, repo)
            print("Crawl commits done")
            print("Num commits before preprocess:", len(commits))
            # Get commit messages and commit descriptions
            commits = [get_commit(commit) for commit in commits if get_commit(commit)]
            messages, descriptions = zip(*commits)
            idx = range(1, len(commits) + 1)
            commits_df = pd.DataFrame({"Index": idx, "Message": messages, "Description": descriptions})
            print("Num commits before remove duplicates:", len(commits_df))  # Add to test
            commits_df = commits_df.drop_duplicates(subset=["Message"]).reset_index(drop=True)
            print("Num commits after remove duplicates:", len(commits_df))  # Add to test
            # Check commit messages
            print("Num commit messages:", len(commits_df))

            print("\n")
            print("==============================================")
            print("\n")

            # Save data to folder
            repo_folder_path = os.path.join(ROOT_DIR, "data", folder_name)
            if not os.path.exists(repo_folder_path):
                os.mkdir(repo_folder_path)

            changelogs_path = os.path.join(ROOT_DIR, "data", folder_name, "changelogs.csv")
            commits_path = os.path.join(ROOT_DIR, "data", folder_name, "commits.csv")
            changelogs_df.to_csv(changelogs_path, index=False)
            commits_df.to_csv(commits_path, index=False)
            repos.loc[i, "Crawl status"] = "Done"
        except:
            repos.loc[i, "Crawl status"] = "Error"
        
        repos.to_csv(repos_path, index=False)
            
    




    


    