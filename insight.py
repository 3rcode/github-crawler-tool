import os
import pandas as pd
import numpy as np          
from settings import ROOT_DIR
import matplotlib.pyplot as plt
import re
from typing import Tuple


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
        

def check_releases_relationship():
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    # Check repos
    print(repos.shape)
    print(repos.head())
    num_repo = repos.shape[0]
    time_between_2_release = np.array([]).astype("timedelta64")
    num_commit_between_2_releases = np.array([])
    outlier_cases = open("outlier_cases.txt", "a+")
    num_changelog = 0
    for i in range(num_repo):
        owner = repos.loc[i, "Owner"]
        repo = repos.loc[i, "Repo"]
        folder = f"{owner}_{repo}"
        print("Repo:", folder)
        path = os.path.join(ROOT_DIR, "data", folder, "changelog_info.csv")
        changelog_info = pd.read_csv(path)
        num_changelog += len(changelog_info)    
        changelog_info = changelog_info.drop(columns=["index"])
        changelog_info["created_at"] = pd.to_datetime(changelog_info["created_at"])
        changelog_info["published_at"] = pd.to_datetime(changelog_info["published_at"])
        changelog_info = changelog_info.sort_values(by=["created_at", "published_at"], ascending=[False, False])
        changelog_info = changelog_info.set_index(np.arange(len(changelog_info.index)))
        repo_path = os.path.join(ROOT_DIR, "..", "repos", folder)

        # Drop releases that have same created_at field (create at the same time) and target to the same commit
        try:
            duplicates = []
            for j in range(len(changelog_info) - 1):
                if changelog_info.loc[j, "created_at"] == changelog_info.loc[j + 1, "created_at"]:
                    commit_cmd_1 = f"""cd {repo_path}
                        git rev-parse {changelog_info.loc[j, "tag_name"]}~"""
                    commit_sha_1 = os.popen(commit_cmd_1).read()

                    commit_cmd_2 = f"""cd {repo_path}
                        git rev-parse {changelog_info.loc[j + 1, "tag_name"]}~"""
                    commit_sha_2 = os.popen(commit_cmd_2).read()
                    
                    tag_cmd_1 = f"""cd {repo_path}
                        git rev-parse {changelog_info.loc[j, "tag_name"]}"""
                    tag_sha_1 = os.popen(tag_cmd_1).read()

                    tag_cmd_2 = f"""cd {repo_path}
                        git rev-parse {changelog_info.loc[j + 1, "tag_name"]}"""    
                    tag_sha_2 = os.popen(commit_cmd_2).read()
                
                    if tag_sha_1 == tag_sha_2:
                        duplicates.append(j + 1)
                    else:
                        tag_1 = changelog_info.loc[j, "tag_name"]
                        tag_2 = changelog_info.loc[j + 1, "tag_name"]
                        case = f"**Outlier at:\tRepo: {folder}**\n"\
                                "```\n"\
                               f"Tag1: {tag_1},sha: {tag_sha_1},Commit: {commit_sha_1}\n"\
                               f"Tag2: {tag_2},sha: {tag_sha_2},Commit: {commit_sha_2}\n"\
                               "```\n"
                        outlier_cases.writelines(case)
            changelog_info = changelog_info.drop(duplicates)
            changelog_info = changelog_info.set_index(np.arange(len(changelog_info.index)))
        except:
            print("False")
        
        # Get time between two consecutive release
        try:
            times = np.array([changelog_info.loc[j, "created_at"] - changelog_info.loc[j + 1, "created_at"]
                              for j in range(len(changelog_info) - 1)]).astype("timedelta64")
            time_between_2_release = np.append(times, time_between_2_release)
        except Exception as e:
            print(e)
            break
        
    outlier_cases.close()
    print("Time between two releases:")
    print("\tMax:", np.max(time_between_2_release))
    print("\tMin:", np.min(time_between_2_release))
    print("\tMean:", np.mean(time_between_2_release))
    print("\tMedian:", np.median(time_between_2_release))
    # print("Num commit between two releases")
    # print("\tMax:", np.max(num_commit_between_2_releases))
    # print("\tMin:", np.min(num_commit_between_2_releases))
    # print("\tMean:", np.mean(num_commit_between_2_releases))
    # print("\tMedian:", np.median(num_commit_between_2_releases))

        
        
    
check_releases_relationship()