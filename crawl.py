import requests
import pandas as pd
import os
from settings import HEADERS, ROOT_DIR
from bs4 import BeautifulSoup
from markdown import markdown


def crawl_changelogs(owner, repo):
    page = 1
    all_changelogs = []
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/releases?per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        changelogs = response.json()
        changelogs_per_page = [changelog["body"] for changelog  in changelogs]
        all_changelogs += changelogs_per_page
        if len(changelogs) < 100:
            break
        page += 1
    return all_changelogs

def crawl_commits(owner, repo):
    page = 1
    all_commits = []
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        commits = response.json()
        commits_per_page = [commit["commit"]["message"] for commit in commits]
        all_commits += commits_per_page
        if len(commits) < 100:
            break
        page += 1
    return all_commits

def get_commit(commit, replace=False):
    if not commit:
        return None
    html = markdown(commit)
    soup = BeautifulSoup(html, "html.parser")
    sentences = [p.text.strip() for p in soup.find_all('p')]
    if not sentences:
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
            changelogs_df = changelogs_df.drop_duplicates().reset_index(drop=True)

            # Check changelog sentences
            print("Num changelog sentences:", len(changelogs_df))

            # Crawl commits
            print("Start crawl commits")
            commits = crawl_commits(owner, repo)
            print('Crawl commits done')

            # Get commit messages and commit descriptions
            commits = [get_commit(commit) for commit in commits if get_commit(commit)]
            messages, descriptions = zip(*commits)
            idx = range(1, len(commits) + 1)
            commits_df = pd.DataFrame({"Index": idx, "Message": messages, "Description": descriptions})
            commits_df = commits_df.drop_duplicates(subset=["Message"]).reset_index(drop=True)
            
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
            
    




    


    