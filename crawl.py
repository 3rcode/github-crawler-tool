import requests
from markdown import markdown
from bs4 import BeautifulSoup

github_token = "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"
headers = {
    'Authorization': f'token {github_token}'
}


def crawl_releases(username, repo_name):
    page = 1
    all_releases = []
    while True:
        url = f"https://api.github.com/repos/{username}/{repo_name}/releases?per_page=100&page={page}"
        response = requests.get(url, headers=headers)
        releases = response.json()
        releases_per_page = list(map(lambda x: [x["tag_name"], x["body"]], releases))
        all_releases += releases_per_page
        if len(releases_per_page) < 100:
            break
        page += 1
    return all_releases


def crawl_compare_commit(username, repo_name, releases):
    compare_commits = []
    for i in range(len(releases) - 1):
        release_commit = []
        page = 1
        while True:
            url = f"https://api.github.com/repos/{username}/{repo_name}/compare/" \
                  f"{releases[i + 1][0]}...{releases[i][0]}?per_page=250&page={page}"
            try:
                response = requests.get(url, headers=headers)
                compare_commit = response.json()
                compare_commit = compare_commit["commits"]
                compare_commit = list(map(lambda x: BeautifulSoup(markdown(x["commit"]["message"]), "html.parser").get_text(),
                                          compare_commit))
                release_commit += compare_commit
            except Exception as e:
                print(e)

            if len(compare_commit) < 100:
                break
            page += 1
        releases[i][1] = BeautifulSoup(markdown(releases[i][1]), "html.parser").get_text()
        compare_commits.append([releases[i][0], releases[i][1], release_commit])
    return compare_commits
