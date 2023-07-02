import requests
import pandas as pd
import os
import traceback
from bs4 import BeautifulSoup
from markdown import markdown
import regex as re


github_token = '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'
headers = {
    'Authorization': f'token {github_token}',   
    'Accept': 'application/vnd.github.v3+json'
}

def crawl_changelogs(owner, repo):
    page = 1
    all_changelogs = []
    while True:
        url = f'https://api.github.com/repos/{owner}/{repo}/releases?per_page=100&page={page}'
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        changelogs = response.json()
        changelogs_per_page = [changelog['body'] for changelog  in changelogs]
        all_changelogs += changelogs_per_page
        if len(changelogs) < 100:
            break
        page += 1
    return all_changelogs

def crawl_commits(owner, repo):
    page = 1
    all_commits = []
    while True:
        url = f'https://api.github.com/repos/{owner}/{repo}/commits?per_page=100&page={page}'
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        commits = response.json()
        commits_per_page = [commit['commit']['message'] for commit in commits]
        all_commits += commits_per_page
        if len(commits) < 100:
            break
        page += 1
    return all_commits

def get_commit(commit, replace=False):
    if not commit:
        return None
    html = markdown(commit)
    soup = BeautifulSoup(html, 'html.parser')
    sentences = [p.text.strip() for p in soup.find_all('p')]
    if not sentences:
        return None
    message = sentences[0]
    description = sentences[1:]
    description = '<.> '.join(description)
    return message, description

def get_changelog_sentences(changelog, replace=False):
    if not changelog:
        return []
    html = markdown(changelog)
    soup = BeautifulSoup(html, 'html.parser')
    sentences = [li.text.strip() for li in soup.find_all('li')]
    return sentences

def crawl_data():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(ROOT_DIR, 'data', 'Corpus Repo - Training.csv')
    corpus_repo_training = pd.read_csv(corpus_path)
    
    # Check corpus repo training
    print(corpus_repo_training.shape)
    print(corpus_repo_training.head())
    
    num_repo = corpus_repo_training.shape[0]

    for i in range(num_repo):
        owner = corpus_repo_training.loc[i, 'User']
        repo = corpus_repo_training.loc[i, 'Repo name']
        folder_name = f'{owner}_{repo}'
        print(owner, repo)
        try:
            # Crawl changelogs
            changelogs = crawl_changelogs(owner, repo)
            print('Crawl changelogs done')

            # Get changelogs sentences
            changelogs_sentences = [get_changelog_sentences(changelog) for changelog in changelogs]
            changelogs_sentences = [sentence for changelog_sentences in changelogs_sentences
                                            for sentence in changelog_sentences ]
            idx = range(1, len(changelogs_sentences) + 1)
            changelogs_df = pd.DataFrame({'Index': idx, 'Changelog Sentence': changelogs_sentences})
            print(changelogs_df.head())
            print(changelogs_df.shape)
            # Crawl commits
            commits = crawl_commits(owner, repo)
            print('Crawl commits done')

            # Get commit messages and commit descriptions
            commits = [get_commit(commit) for commit in commits if get_commit(commit)]
            messages, descriptions = zip(*commits)
            idx = range(1, len(commits) + 1)
            commits_df = pd.DataFrame({'Index': idx, 'Message': messages, 'Description': descriptions})

            # Check dataframe
            print(commits_df.shape)
            print(changelogs_df.shape)
            print(commits_df.head())
            print(changelogs_df.head())


            # Save data to folder
            repo_folder_path = os.path.join(ROOT_DIR, 'data', folder_name)
            if not os.path.exists(repo_folder_path):
                os.mkdir(repo_folder_path)

            changelogs_path = os.path.join(ROOT_DIR, 'data', folder_name, 'changelogs.csv')
            commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'commits.csv')
            changelogs_df.to_csv(changelogs_path, index=False)
            commits_df.to_csv(commits_path, index=False)
            corpus_repo_training.loc[i, 'Crawl status'] = 'Done'
        except:
            corpus_repo_training.loc[i, 'Crawl status'] = 'Error'
        
        corpus_repo_training.to_csv(corpus_path, index=False)

def is_commit(href):
    components = href.split('/')
    return (len(components) == 7 and components[0] == 'https:' and components[1] == ''
            and components[2] == 'github.com' and components[5] == 'commit')

def is_pull(href):
    components = href.split('/')
    return (len(components) == 7 and components[0] == 'https:' and components[1] == ''
            and components[2] == 'github.com' and components[5] == 'pull')

def sha(commit_link):
    components = commit_link.split('/')
    return components[6]

def pull_number(pull_link):
    components = pull_link.split('/')
    return components[6]

def collect_hash_link(changelog):
    if not changelog:
        return ([], [])
    html = markdown(changelog)
    
    soup = BeautifulSoup(html, 'html.parser')
    commit_shas = list()
    pull_numbers = list()
    for li in soup.find_all('li'):
        for a in li.find_all('a'):
            href = a.get('href')
            if is_commit(href):
                commit_shas.append(sha(href))
            if is_pull(href):
                pull_numbers.append(pull_number(href))
    return (commit_shas, pull_numbers)

def crawl_commit_from_sha(owner, repo, _sha):
    url = f'https://api.github.com/repos/{owner}/{repo}/commits/{_sha}'
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
    commit = response.json()
    return commit['commit']['message']

def crawl_compare_commit(owner, repo, base, head):
    url = f'https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}'
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
    commits = response.json()['commits']
    all_commits = [commit['commit']['message'] for commit in commits]
    return all_commits

def crawl_commits_from_pull_number(owner, repo, _pull_number):
    pull_request_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{_pull_number}'
    response = requests.get(pull_request_url, headers=headers)
    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
    pull_request = response.json()
    merge_commit_sha = pull_request['merge_commit_sha']
    head_commit_sha = pull_request['head']['sha']
    base_commit_sha = pull_request['base']['sha']
    merge_commit = crawl_commit_from_sha(owner, repo, merge_commit_sha)
    compare_commits = crawl_compare_commit(owner, repo, base_commit_sha, head_commit_sha)
    return [merge_commit, *compare_commits]
    
def assess_label_data_method():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(ROOT_DIR, 'data', 'Corpus Repo - Training.csv')
    corpus_repo_training = pd.read_csv(corpus_path)
    
    repos_testing = corpus_repo_training.sample(frac=0.4, random_state=1).reset_index(drop=True)
    print(repos_testing.shape)
    num_repo = repos_testing.shape[0]

    for i in range(num_repo):
        owner = repos_testing.loc[i, 'User']
        repo = repos_testing.loc[i, 'Repo name']
        print(owner, repo)
        changelogs = crawl_changelogs(owner, repo)
        all_commit_shas = set()
        all_pull_numbers = set()
        print(len(changelogs))
        for changelog in changelogs:
            try:
                commit_shas, pull_numbers = collect_hash_link(changelog)
                all_commit_shas.update(commit_shas)
                all_pull_numbers.update(pull_numbers)
            except:
                print('Terminate')
                break
        all_commits = set()
        all_commits.update([crawl_commit_from_sha(owner, repo, _sha) for _sha in all_commit_shas])
        for _pull_number in all_pull_numbers:
            all_commits.update(crawl_commits_from_pull_number(owner, repo, _pull_number))
        for commit in all_commits:
            print(commit)
        break   
            
if __name__ == '__main__':
    # crawl_data()
    assess_label_data_method()
            
    




    


    