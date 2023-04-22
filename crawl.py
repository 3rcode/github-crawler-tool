import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
from markdown import markdown
import re

owner = 'revoltchat'
repo = 'desktop'
github_token = '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'
headers = {
    'Authorization': f'token {github_token}',
    'Accept': 'application/vnd.github.v3+json'
}

def crawl_release_notes():
    global headers, owner, repo
    page = 1
    all_releases = []
    while True:
        url = f'https://api.github.com/repos/{owner}/{repo}/releases?per_page=100&page={page}'
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        releases = response.json()
        releases_per_page = list(map(lambda release: release['body'], releases))
        all_releases += releases_per_page
        if len(releases) < 100:
            break
        page += 1
    return all_releases


def get_commit_message(commit):
    commit_lines = commit.splitlines()
    message = commit_lines[0]
    description = ' '.join(commit_lines[1:])
    return message, description


def crawl_commit():
    global headers, owner, repo
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
        commits_per_page = list(map(lambda commit: commit['commit']['message'], commits))
        all_commits += commits_per_page
        if len(commits) < 100:
            break
        page += 1
    return all_commits

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(string=True))

    return text


if __name__ == '__main__':
    # Crawl release notes and commits
    release_notes = crawl_release_notes()
    print('Crawl releases done')
    release_notes = [markdown_to_text(release) for release in release_notes]
    idx = range(1, len(release_notes) + 1)
    release_notes_df = pd.DataFrame({'Index': idx, 'Release Note': release_notes})
    
    commits = crawl_commit()
    print('Crawl commits done')
    commits = [markdown_to_text(commit) for commit in commits]
    messages = []
    descriptions = []
    for commit in commits:
        message, description = get_commit_message(commit)
        messages.append(message)
        descriptions.append(description)

    idx = range(1, len(messages) + 1)
    commits_df = pd.DataFrame({'Index': idx, 'Message': messages, 'Description': descriptions})

    # Check dataframe

    # print(commits_df.shape)
    # print(release_notes_df.shape)
    # print(commits_df.head())
    # print(release_notes_df.head())


    # Save data to folder
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO = f'{owner}_{repo}'
    repo_folder_path = os.path.join(ROOT_DIR, REPO)
    os.mkdir(repo_folder_path)
    release_notes_path = os.path.join(ROOT_DIR, REPO, 'release_notes.csv')
    commits_path = os.path.join(ROOT_DIR, REPO, 'commits.csv')
    release_notes_df.to_csv(release_notes_path, index=False)
    commits_df.to_csv(commits_path, index=False)


    


    