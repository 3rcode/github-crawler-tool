import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
from markdown import markdown
import regex as re
import numpy as np


github_token = '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'
headers = {
    'Authorization': f'token {github_token}',   
    'Accept': 'application/vnd.github.v3+json'
}

def crawl_changelogs(owner, repo):
    global headers
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


def crawl_commit(owner, repo):
    global headers
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
        return ('', [])
    html = markdown(commit)
    soup = BeautifulSoup(html, 'html.parser')
    sentences = [p.text.stip() for p in soup.find_all('p')]
    message = sentences[0]
    description = sentences[1:]
    description = '<.> '.join(description)
    return message, description

def get_release_note_sentences(release, replace=False):
    if not release:
        return []
    html = markdown(release)
    soup = BeautifulSoup(html, 'html.parser')
    sentences = [li.text.strip() for li in soup.find_all('li')]
    # print(sentences)
    return sentences


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """
    if not markdown_string:
        return ''
    # md -> html -> text 
    html = markdown(markdown_string)
    soup = BeautifulSoup(html, "html.parser")
    # Extract text    
    text = ''.join(soup.findAll(string=True))
    return text

def markdown_to_text_abstract(markdown_string):
    # md -> html -> text 
    if not markdown_string:
        return ''
    # Replace link with 'link' as abstraction
    markdown_string = re.sub(r'(?|(?<txt>(?<url>(?:ht|f)tps?://\S+(?<=\P{P})))|\(([^)]+)\)\[(\g<url>)\])', 'link', markdown_string)
    
    html = markdown(markdown_string)

    soup = BeautifulSoup(html, "html.parser")

    # Replace code tag with 'module' as abstraction
    for code in soup.find_all('code'):
        code.string = 'module'

    # Extract text
    text = ''.join(soup.findAll(string=True))
    return text

if __name__ == '__main__':
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

        # Crawl release notes and commits
        release_notes = crawl_release_notes(owner, repo)
        print('Crawl releases done')

        # Get release note sentences
        release_notes_sentences = np.array([get_release_note_sentences(release_note) 
                                            for release_note in release_notes])
        release_notes_sentences = release_notes_sentences.flatten().reshape(-1, 1)
        idx = np.array(range(1, len(release_notes) + 1))
        
        release_notes_df = pd.DataFrame(np.hstack((idx, release_notes_sentences)), 
                                        columns=['Index', 'Release Note'])
        
        # Crawl commits
        commits = crawl_commit(owner, repo)
        print('Crawl commits done')

        # Get commit messages and commit descriptions
        commits = [get_commit_message(commit) for commit in commits]
        messages, descriptions = zip(*commits)
        idx = range(1, len(commits) + 1)
        commits_df = pd.DataFrame({'Index': idx, 'Message': messages, 'Description': descriptions})

        # Check dataframe
        print(commits_df.shape)
        print(release_notes_df.shape)
        print(commits_df.head())
        print(release_notes_df.head())


        # Save data to folder
        repo_folder_path = os.path.join(ROOT_DIR, 'data', folder_name)
        if not os.path.exists(repo_folder_path):
            os.mkdir(repo_folder_path)

        release_notes_path = os.path.join(ROOT_DIR, 'data', folder_name, 'release_notes.csv')
        commits_path = os.path.join(ROOT_DIR, 'data', folder_name, 'commits.csv')
        release_notes_df.to_csv(release_notes_path, index=False)
        commits_df.to_csv(commits_path, index=False)

        corpus_repo_training.loc[i, 'Crawl status'] = 'Done'
        corpus_repo_training.to_csv(corpus_path, index=False)
    




    


    