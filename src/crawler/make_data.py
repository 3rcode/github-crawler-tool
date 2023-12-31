import os
from configparser import ConfigParser
from typing import List, TypeVar

import pandas as pd
import pygit2
import requests

config = ConfigParser()
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
config.read(config_file)

root = config["Paths"]["root"]
repos_store = config["Paths"]["repos_store"]
headers = eval(config["API"]["headers"])


Time = TypeVar("Time")


class MyRemoteCallbacks(pygit2.RemoteCallbacks):
    def transfer_progress(self, stats):
        print(f"{stats.indexed_objects}/{stats.total_objects}")


class MakeData:
    """MakeData class"""

    def __init__(self, owner: str, repo: str):
        """Constructor of the class

        Args:
            owner (str): Github user that is owner of repo
            repo (str): Repo that want to collect data
        """

        self.owner = owner
        self.repo = repo
        self.folder = f"{self.owner}_{self.repo}"

    def _github_api(self, comp: str, option: str = "") -> List[str]:
        """Use github api to collect data

        Args:
            comp (str): Component want to collect
            option (str): Addition query params. Default: ""

        Returns:
            Infomation about componnent of Github repository
        """

        page = 1
        all_els = []
        while True:
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/{comp}?{option}&per_page=100&page={page}"
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(response.status_code)
                print(response.text)
                break
            els = response.json()
            all_els += els
            # 100 is the limit of per_page param in github api
            if len(els) < 100:
                break
            page += 1

        return all_els

    def crawl_release_notes(self):
        """Crawl all release notes of a Github repo"""

        return self._github_api(comp="releases")

    def store_release_note_info(self) -> None:
        """Store information of release notes of Github repo to data/"""

        try:
            # Crawl release notes
            print("Start crawl release notes")
            release_note_info = self.crawl_release_notes()
            print("Crawl release notes done")
            release_note_info = pd.DataFrame(release_note_info)
            print(release_note_info.head())
            folder_path = os.path.join(root, "data", self.folder)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            rn_info_path = os.path.join(folder_path, "release_note_info.csv")
            release_note_info.to_csv(rn_info_path)
        except Exception as e:
            raise e

    def crawl_branches(self):
        """Crawl all branches of a Github repo"""

        return self._github_api(comp="branches")

    def crawl_commits_using_api(self):
        """Crawl all commits of a Github repo using Github API"""

        branches = [branch["name"] for branch in self.crawl_branches()]
        print("Num branches:", len(branches))
        all_commits = []
        for branch in branches:
            commits = self._github_api(comp="commits", option=f"sha={branch}")
            print(f"Number commits in branch {branch}:", len(commits))
            all_commits.extend(commits)

        return all_commits

    def crawl_commits_after_clone(self) -> List[str]:
        """Get all commits of Github repo using method that clone repository and using git command to
        retrieval"""
        self.clone_repos()
        path = os.path.join(repos_store, self.folder)
        print(path)
        cmd = f""" cd {path}
                git branch -a"""
        all_branches = os.popen(cmd).read().split("\n")[:-1]
        all_branches = [branch.strip() for branch in all_branches]
        all_branches.remove("* main")
        remote_branches = [
            "/".join(branch.split("/")[2:]) for branch in all_branches[1:]
        ]
        remote_branches.remove("HEAD -> origin/main")
        all_commit_shas = set()
        for branch in all_branches:
            try:
                if branch in remote_branches:
                    cmd = f"""cd {path}
                    git rev-list remotes/origin/{branch}"""
                else:
                    cmd = f"""cd {path}
                    git rev-list {branch}"""
                commit_shas = os.popen(cmd).read()
                # Each line is a commit sha and the last line is empty line
                commit_shas = commit_shas.split("\n")[:-1]
                all_commit_shas.update(commit_shas)
            except:
                continue

        repo = pygit2.Repository(path)
        # Get commit message from commit sha
        commits = [repo.revparse_single(commit_sha) for commit_sha in all_commit_shas]
        commits = [
            {
                "message": commit.message,
                "sha": commit.hex,
                "author": commit.author,
                "commit_time": commit.commit_time,
                "committer": commit.committer,
            }
            for commit in commits
        ]

        return commits

    def clone_repos(self) -> None:
        """Clone github repository"""

        path = os.path.join(repos_store, self.folder)
        if not os.path.exists(path):
            pygit2.clone_repository(
                f"https://github.com/{self.owner}/{self.repo}",
                path,
                callbacks=MyRemoteCallbacks(),
            )
