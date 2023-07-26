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
import datetime

def commit_to_release(repo_path: str, sorted_releases: pd.DataFrame, outlier_path: str) -> List[int]:
    print("dang chay")
    times = []
    outlier = open(outlier_path, "a+")
    for i in range(len(sorted_releases) - 1):
        cmd = f"""cd {repo_path}
        git rev-list --reverse {sorted_releases.loc[i, "tag_name"]} ^{sorted_releases.loc[i + 1, "tag_name"]}"""
        sha = os.popen(cmd).read().split('\n')

        print(cmd)
        if sha:
            last_sha = sha[0]
            cmd = f"""cd {repo_path}
            git show --no-patch --no-notes --pretty="%ct" {last_sha}"""
            last_commit_time = int(os.popen(cmd).read())
            time = sorted_releases.loc[i, "created_at"].to_pydatetime().timestamp() - last_commit_time
            if time <= 0: 
                outlier.writelines(f"Outlier at: {repo_path}\t{sorted_releases.loc[i, 'tag_name']}\n")
            else:
                print(str(datetime.timedelta(seconds=time)))
                times.append(time)
    outlier.close()

    return times


def check_releases_relationship():
    print("Here")
    repos_path = os.path.join(ROOT_DIR, "data", "Repos.csv")
    repos = pd.read_csv(repos_path)
    # Check repos
    print(repos.shape)
    print(repos.head())
    num_repo = repos.shape[0]
    time_between_2_release = np.array([]).astype("timedelta64")
    time_commit_to_release = np.array([])
    num_commit_between_2_releases = np.array([]).astype("int64")
    outlier_cases = "outlier_cases.txt"
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
        # changelog_info = changelog_info.drop_duplicates(subset=["created_at"]).reset_index(drop=True)
        # changelog_info = changelog_info.drop_duplicates(subset=["commit_sha"]).reset_index(drop=True)
        changelog_info = changelog_info.set_index(np.arange(len(changelog_info.index)))
        # Get time between two consecutive release
        try:
            # times = np.array([changelog_info.loc[j, "created_at"] - changelog_info.loc[j + 1, "created_at"]
            #                   for j in range(len(changelog_info) - 1)]).astype("timedelta64")
            # time_between_2_release = np.append(times, time_between_2_release)
            # num_commit_between_2_releases = np.append(
            #     np.array(commit_between_two_releases(repo_path, changelog_info, outlier_cases)).astype("int64"),
            #     num_commit_between_2_releases
            # )
            time_commit_to_release = np.append(
                np.array(commit_to_release(repo_path, changelog_info,outlier_cases)),
                time_commit_to_release
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
    # print("Time between two releases:")
    # print("\tMax:", np.max(time_between_2_release))
    # print("\tMin:", np.min(time_between_2_release))
    # print("\tMean:", np.mean(time_between_2_release))
    # print("\tMedian:", np.median(time_between_2_release))
    # df = pd.DataFrame(time_between_2_release, columns=["Time"])
    # df["Time"] = df["Time"].dt.total_seconds().astype("int")
    # df.to_csv("time_between_2_re.csv")
    # num_commit_between_2_releases = num_commit_between_2_releases[num_commit_between_2_releases != 0]
    
    # print("Num commit between two releases")
    # print("\tMax:", np.max(num_commit_between_2_releases))
    # print("\tMin:", np.min(num_commit_between_2_releases))
    # print("\tMean:", np.mean(num_commit_between_2_releases))
    # print("\tMedian:", np.median(num_commit_between_2_releases))
    # pd.DataFrame(num_commit_between_2_releases, columns=["Num commit"]).to_csv("num_cm_between_2_re.csv")

check_releases_relationship()
