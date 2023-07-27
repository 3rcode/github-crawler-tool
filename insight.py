import os
import pandas as pd
import numpy as np          
from settings import ROOT_DIR, HEADERS
import re
import requests
from typing import Tuple, List
from make_data import github_api
from markdown import markdown
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime

def summarize_data() -> None:
    """ Summarize number of commits and number of changelog sentences """

    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    data = []
    for i in range(len(repos)):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder = f"{owner}_{repo}"
        changelog_sen_path = os.path.join(ROOT_DIR, "data", folder, "changelog_sentence.csv")
        commit_path = os.path.join(ROOT_DIR, "data", folder, "commit.csv")
        

        # Load raw data
        changelog_sen_df = pd.read_csv(changelog_sen_path)
        commit_df = pd.read_csv(commit_path)
        num_changelog_sen = len(changelog_sen_df)
        num_commit = len(commit_df)

        data.append([repo, num_commit, num_changelog_sen])        
        

    data = pd.DataFrame(data, columns=["Repo", "Num Commit", "Num Changelog Sentence"])
    data.set_index("Repo")
    data.to_csv("data_info.csv", index=False)


def check_bot() -> Tuple[float, float]:
    """ Check whether author and commiter is bot """

    path = os.path.join(ROOT_DIR, "data", "all_data.csv")
    all_data = pd.read_csv(path)
    print(all_data.info())
    pattern_author_bot = r".*\[bot\].*"
    num_author_bot = sum([1 if re.search(pattern_author_bot, author) else 0 for author in all_data.loc[:, "Author"]])
    num_github_committer = len(all_data[all_data["Committer"] == "GitHub <noreply@github.com>"])
    
    return num_author_bot / len(all_data), num_github_committer / len(all_data)


def check_commit_not_in_release_branches() -> float:
    """ Crawl commits and changelog sentences of all repositories in "Repos.csv" file """
    
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    # Check repos
    print(repos.shape)
    print(repos.head())
    num_repo = repos.shape[0]
    commit_in_release_branch = all_commit = 0

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
            # Get branches that had had release version
            target_branches = changelog_info["target_commitish"].unique().tolist()

            # Defint path to folder of cloned repo
            path = os.path.join(ROOT_DIR, "..", "repos", f"{owner}_{repo}")
            
            # Count number of commit in branches that had had realease version
            cmd = f""" cd {path} 
                    git branch -a"""
            all_branches = os.popen(cmd).read().split('\n')[:-1]
            all_branches = [branch.strip() for branch in all_branches]
            remote_branches = ['/'.join(branch.split('/')[2:]) for branch in all_branches[1:]]
            target_branch_shas = set()
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
                    target_branch_shas.update(commit_shas)
                except:
                    continue
            num_target_branch_commit = len(target_branch_shas)
            commit_in_release_branch += num_target_branch_commit

            # Count number of all commits in all branches
            cmd = f"""cd {path}
                git rev-list --branches=* --remotes=* --count"""
            num_commit = int(os.popen(cmd).read())
            all_commit += num_commit
        except Exception as e:
            print(e)

    return 1 - commit_in_release_branch / num_commit 


def commit_between_two_releases(repo_path: str, sorted_releases: pd.DataFrame, outlier_path: str) -> List[int]:
    """ Calculate number of commits between two releases version of a Github repo """
    
    compare_commit = []
    outlier = open(outlier_path, "a+")
    for i in range(len(sorted_releases) - 1):
        try:
            for j in range(i + 1, len(sorted_releases)):
                cmd = f"""cd {repo_path}
                    git rev-list {sorted_releases.loc[j, "tag_name"]}..{sorted_releases.loc[i, "tag_name"]} --count"""
                num_commit = int(os.popen(cmd).read())
                if num_commit > 0:
                    compare_commit.append(num_commit)
            else:
                cmd = f"""cd {repo_path}
                    git rev-list {sorted_releases.loc[i, "tag_name"]} --count"""
                num_commit = int(os.popen(cmd).read())
                if num_commit > 0:
                    compare_commit.append(num_commit)
        except Exception as e:
            outlier.write(f"{repo_path}\t{sorted_releases.loc[i, 'tag_name']}" \
                          f"\t{sorted_releases.loc[j, 'tag_name']}\n")

    outlier.close()
    
    return compare_commit


def time_commit_to_release(repo_path: str, sorted_releases: pd.DataFrame, outlier_path: str) -> List[int]:
    """ Calculate time from oldest commit of a release version to that release public """

    times = []
    outlier = open(outlier_path, "a+")
    for i in range(len(sorted_releases) - 1):
        cmd = f"""cd {repo_path}
        git rev-list --reverse {sorted_releases.loc[i, "tag_name"]} ^{sorted_releases.loc[i + 1, "tag_name"]}"""
        sha = os.popen(cmd).read().split('\n')
        if sha:
            last_sha = sha[0]
            cmd = f"""cd {repo_path}
            git show --no-patch --no-notes --pretty="%ct" {last_sha}"""
            last_commit_time = int(os.popen(cmd).read())
            time = sorted_releases.loc[i, "created_at"].to_pydatetime().timestamp() - last_commit_time
            if time <= 0: 
                outlier.writelines(f"Outlier at: {repo_path}\t{sorted_releases.loc[i, 'tag_name']}\n")
            else:  
                times.append(time)
    outlier.close()

    return times


def check_releases_relationship() -> None:
    """ Do some functions with sorted releases"""

    # Load repositories
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    # Check repos
    print(repos.shape)
    print(repos.head())

    # Initial 
    time_between_2_release = np.array([]).astype("timedelta64")
    time_commit_2_release = np.array([])
    num_commit_between_2_releases = np.array([]).astype("int64")
    outlier_cases = "outlier_cases.txt"

    # Traverse all repositories
    for i in range(len(repos)):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder = f"{owner}_{repo}"
        print("Repo:", folder)
        path = os.path.join(ROOT_DIR, "data", folder, "changelog_info.csv")
        changelog_info = pd.read_csv(path)

        # Sort release version by created and published time
        changelog_info = changelog_info.drop(columns=["index"])
        changelog_info["created_at"] = pd.to_datetime(changelog_info["created_at"])
        changelog_info["published_at"] = pd.to_datetime(changelog_info["published_at"])
        changelog_info = changelog_info.sort_values(by=["created_at", "published_at"], ascending=[False, False])
        changelog_info = changelog_info.set_index(np.arange(len(changelog_info.index)))
        repo_path = os.path.join(ROOT_DIR, "..", "repos", folder)

        # Add tag sha and commit sha corresponding with release tag name
        tag_shas = []
        commit_shas = []
        for j in range(len(changelog_info)):
            tag_cmd = f"""cd {repo_path}
                    git rev-parse {changelog_info.loc[j, "tag_name"]}"""
            tag_sha = os.popen(tag_cmd).read()
            tag_shas.append(tag_sha[:-1])
            commit_cmd = f"""cd {repo_path}
                        git rev-parse {changelog_info.loc[j, "tag_name"]}~"""
            commit_sha = os.popen(commit_cmd).read() 
            commit_shas.append(commit_sha[:-1])       
        changelog_info["tag_sha"] = tag_shas
        changelog_info["commit_sha"] = commit_shas 

        # Remove releases that created at the same time or pointed to the same commit
        changelog_info = changelog_info.drop_duplicates(subset=["created_at"]).reset_index(drop=True)
        changelog_info = changelog_info.drop_duplicates(subset=["commit_sha"]).reset_index(drop=True)
        changelog_info = changelog_info.set_index(np.arange(len(changelog_info.index)))
        # Get time between two consecutive release
        try:
            # Find all time between two consecutive releases
            times = np.array([changelog_info.loc[j, "created_at"] - changelog_info.loc[j + 1, "created_at"]
                              for j in range(len(changelog_info) - 1)]).astype("timedelta64")
            time_between_2_release = np.append(times, time_between_2_release)

            # Find number of commits between two releases
            num_commit_between_2_releases = np.append(
                np.array(commit_between_two_releases(repo_path, changelog_info, outlier_cases)).astype("int64"),
                num_commit_between_2_releases
            )

            # Find time from oldest commit of a release version to the time that release created
            time_commit_2_release = np.append(
                np.array(time_commit_to_release(repo_path, changelog_info,outlier_cases)),
                time_commit_2_release
            )
        except Exception as e:
            print(e)
            break
    
    print("Time last commit to release:")
    print("\tMax:", f"{np.max(time_commit_to_release)}s")
    print("\tMin:", f"{np.min(time_commit_to_release)}s")
    print("\tMean:", f"{np.mean(time_commit_to_release)}s")
    print("\tMedian:", f"{np.median(time_commit_to_release)}s")
    pd.DataFrame(time_commit_to_release, columns=["Time"]).to_csv("time_cm_2_re.csv")

    print("Time between two releases:")
    print("\tMax:", np.max(time_between_2_release))
    print("\tMin:", np.min(time_between_2_release))
    print("\tMean:", np.mean(time_between_2_release))
    print("\tMedian:", np.median(time_between_2_release))
    df = pd.DataFrame(time_between_2_release, columns=["Time"])
    df["Time"] = df["Time"].dt.total_seconds().astype("int")
    df.to_csv("time_between_2_re.csv")

    num_commit_between_2_releases = num_commit_between_2_releases[num_commit_between_2_releases != 0]
    print("Num commit between two releases")
    print("\tMax:", np.max(num_commit_between_2_releases))
    print("\tMin:", np.min(num_commit_between_2_releases))
    print("\tMean:", np.mean(num_commit_between_2_releases))
    print("\tMedian:", np.median(num_commit_between_2_releases))
    pd.DataFrame(num_commit_between_2_releases, columns=["Num commit"]).to_csv("num_cm_between_2_re.csv")

     
def check_number_of_author() -> None:
    """ Check number of authors in a Github repo"""

    repo_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repo_path)
    all_authors = set()
    for i in range(len(repos)):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i,  "Repo"]
        path = os.path.join(ROOT_DIR, "..", "repos", f"{owner}_{repo}")
        cmd = f"""cd {path}
            git shortlog -s -n -e --all"""
        try:
            authors = os.popen(cmd).read().split("\n")[:-1]
            authors = [author[7:] for author in authors]
            all_authors.update(authors)
        except:
            print(owner, repo)

    print(len(all_authors)) 
    num_authors = np.array(num_authors)
    print("Max:", np.max(num_authors))
    print("Min:", np.min(num_authors))
    print("Mean:", np.mean(num_authors))
    print("Median:", np.median(num_authors))


def check_num_language() -> None:
    """ Check programming language Github repo used """

    repo_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repo_path)
    all_languages = set()
    for i in range(len(repos)):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i,  "Repo"]
        url = f"https://api.github.com/repos/{owner}/{repo}/languages"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            break
        languages = response.json()
        all_languages.update(languages.keys())
    print(all_languages)
    print(len(all_languages))
    

def check_changelog_has_commit_link(owner: str, repo: str) -> Tuple[int, int]:
    """ Crawl all changelogs in https://github.com/owner/repo that have link to commit"""
    
    func = lambda el: el["body"]
    changelogs = github_api(owner, repo, type="releases", func=func)
    num = 0
    pattern = fr"https:\/\/github\.com\/{owner}\/{repo}\/commit\/[A-Fa-f0-9]+"
    for changelog in changelogs:
        try:
            html = markdown(changelog)
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()
            if re.findall(pattern,text):
                num += 1
        except:
            print(f"Wrong at {owner}/{repo}")
        
    return len(changelogs), num


def crawl_changelog_has_commit_link() -> None:
    """ Traverse all repositories in Repos.csv file to find number of changelog has commit link """
    
    repo_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repo_path)
    total_changelog = total_changelog_has_link = 0
    for i in range(len(repos)):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        print("Repo:", owner, repo)
        num_changelog, num_changelog_has_commit_link = check_changelog_has_commit_link(owner, repo)
        total_changelog += num_changelog
        total_changelog_has_link += num_changelog_has_commit_link
        
    
    print("Total changelog:", total_changelog)
    print("Total changelog has commit link:", total_changelog_has_link)
    print("Changelog don't have link to commit:", 
          f"{(total_changelog - total_changelog_has_link) / total_changelog * 100}%")




