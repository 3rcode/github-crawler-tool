import os
import pandas as pd
from markdown import markdown
from bs4 import BeautifulSoup
import requests
from settings import HEADERS, ROOT_DIR
from collections import defaultdict
from crawl import crawl_changelogs, get_commit


err_sha_df = pd.DataFrame({"Owner": [], "Repo": [], "Error Sha": []})
err_pull_num_df = pd.DataFrame({"Owner": [], "Repo": [], "Error Pull Number": []})

def is_commit(href):
    components = href.split('/')
    return (len(components) == 7 and components[0] == "https:" and components[1] == ''
            and components[2] == "github.com" and components[5] == "commit")

def is_pull(href):
    components = href.split('/')
    return (len(components) == 7 and components[0] == "https:" and components[1] == ''
            and components[2] == "github.com" and components[5] == "pull")

def sha(commit_link):
    components = commit_link.split('/')
    return components[6]

def pull_number(pull_link):
    components = pull_link.split('/')
    return components[6]

def collect_refer_commit_link(changelog):
    if not changelog:
        return ([], [])
    html = markdown(changelog)
    soup = BeautifulSoup(html, "html.parser")
    commit_shas = set()
    pull_numbers = set()
    for a in soup.find_all('a'):
        href = a.get("href")
        if is_commit(href):
            commit_shas.add(sha(href))
        if is_pull(href):
            pull_numbers.add(pull_number(href))
    return (commit_shas, pull_numbers)

def crawl_commit_from_sha(owner, repo, _sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{_sha}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        err_sha_df.loc[len(err_sha_df)] = [owner, repo, _sha]
        return None
    commit = response.json()
    return commit["commit"]["message"]

def crawl_compare_commit(owner, repo, base, head):
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception
    compare_commits = response.json()["commits"]
    commits = [commit["commit"]["message"] for commit in compare_commits]
    return commits

def crawl_commits_from_pull_number(owner, repo, _pull_number):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{_pull_number}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        err_pull_num_df.loc[len(err_pull_num_df)] = [owner, repo, _pull_number]
        return []
    pull_request = response.json()
    merge_commit_sha = pull_request["merge_commit_sha"]
    head_commit_sha = pull_request["head"]["sha"]
    base_commit_sha = pull_request["base"]["sha"]
    merge_commit = crawl_commit_from_sha(owner, repo, merge_commit_sha)
    try:
        compare_commits = crawl_compare_commit(owner, repo, base_commit_sha, head_commit_sha)
    except Exception:
        err_pull_num_df.loc[len(err_pull_num_df)] = [owner, repo, _pull_number]
        return []
    if not merge_commit:
        err_pull_num_df.loc[len(err_pull_num_df)] = [owner, repo, _pull_number]
        return []
    return [merge_commit, *compare_commits]

def check_accuracy(owner, repo, test_commits):
    folder = f'{owner}_{repo}'
    labelled_commit_path = os.path.join(ROOT_DIR, "data", folder, "labelled_commits.csv")
    labelled_commits = pd.read_csv(labelled_commit_path)
    total_commit = len(labelled_commits)
    labels = defaultdict(int, labelled_commits["Label"].value_counts())
    label_1 = labels[1]
    label_0 = labels[0]
    total_test_commit = len(test_commits)


    print("Repo:", owner, repo)
    print("\tTotal commit:", total_commit)
    print("\tTotal test commit:", total_test_commit)

    test_labelled_commits = labelled_commits.loc[labelled_commits["Commit Message"].isin(test_commits)]
    test_label = defaultdict(int, test_labelled_commits["Label"].value_counts())
    true_label = test_label[1]
    recall = true_label / total_test_commit
    return [owner, repo, total_commit, label_1, label_0, total_test_commit, recall]
    

def assess_label_data_method():
    # Select random repo to test
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    repos = repos[(repos["Crawl status"] == "Done") & (repos["Label status"] == "Done")]
    repos_testing = repos.sample(frac=0.2, random_state=1).reset_index(drop=True)
    print("Test num:", len(repos_testing))
    num_repo = repos_testing.shape[0]
    
    # File to save test results
    path = os.path.join(ROOT_DIR, "test", "check_label_method.csv")
    check_result = pd.read_csv(path)

    for i in range(num_repo):
        owner = repos_testing.loc[i, "Owner"]
        repo = repos_testing.loc[i, "Repo"]
        print("Repo:", owner, repo)
        
        changelogs = crawl_changelogs(owner, repo)
        all_commit_shas = set()
        all_pull_numbers = set()
        print("Num changelogs:", len(changelogs))
        for changelog in changelogs:
            commit_shas, pull_numbers = collect_refer_commit_link(changelog)
            all_commit_shas.update(commit_shas)
            all_pull_numbers.update(pull_numbers)
            
        print("Num commit shas:", len(all_commit_shas))
        print("Num pull numbers:", len(all_pull_numbers))
        commits = set()
        commits.update([crawl_commit_from_sha(owner, repo, _sha) 
                            for _sha in all_commit_shas if crawl_commit_from_sha(owner, repo, _sha)])
        for _pull_number in all_pull_numbers:
            commit_arr = crawl_commits_from_pull_number(owner, repo, _pull_number)
            if commit_arr:
                commits.update(commit_arr)

        commits = [get_commit(commit) for commit in commits if get_commit(commit)]
        print("Num commit had referred in changelog:", len(commits))
        if commits:
            commit_mes, commit_des = zip(*commits)
            idx = range(1, len(commits) + 1)
            df = pd.DataFrame({"Index": idx, "Commit Message": commit_mes, 
                                "Commit Description": commit_des})
            test_commit_path = os.path.join(ROOT_DIR, "test", "test_commit", f"{owner}_{repo}.csv")
            df.to_csv(test_commit_path, index=False)
            result = check_accuracy(owner, repo, commit_mes)
            check_result.loc[len(check_result)] = result
            check_result.to_csv(path, index=False)
    
    err_sha_path = os.path.join(ROOT_DIR, "test", "error_sha.csv")
    err_pull_num_path = os.path.join(ROOT_DIR, "test", "error_pull_num.csv")
    err_sha_df.to_csv(err_sha_path, index=False)
    err_pull_num_df.to_csv(err_pull_num_path, index=False)
        

