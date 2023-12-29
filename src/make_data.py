import os
from typing import List, TypeVar

import pandas as pd
import pygit2
import requests

from config import HEADERS, REPOS_STORE, ROOT_DIR

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
            response = requests.get(url, headers=HEADERS)
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
            folder_path = os.path.join(ROOT_DIR, "data", self.folder)
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

        branches = [
            branch["name"] for branch in self.crawl_branches(self.owner, self.repo)
        ]
        print("Num branches:", len(branches))
        all_commits = set()
        for branch in branches:
            commits = self.github_api(comp="commits", option=f"sha={branch}")
            print(f"Number commits in branch {branch}:", len(commits))
            all_commits.update(commits)

        return all_commits

    def crawl_commits_after_clone(
        self, target_branches: List[str], lastest: Time, oldest: Time
    ) -> List[str]:
        """Get all commits of Github repo using method that clone repository and using git command to
        retrieval"""

        path = os.path.join(REPOS_STORE, self.folder)
        cmd = f""" cd {path} 
                git branch -a"""
        all_branches = os.popen(cmd).read().split("\n")[:-1]
        all_branches = [branch.strip() for branch in all_branches]
        remote_branches = [
            "/".join(branch.split("/")[2:]) for branch in all_branches[1:]
        ]
        all_commit_shas = set()
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
                commit_shas = commit_shas.split("\n")[:-1]
                all_commit_shas.update(commit_shas)
            except:
                continue

        repo = pygit2.Repository(path)
        # Get commit message from commit sha
        commits = [repo.revparse_single(commit_sha) for commit_sha in all_commit_shas]

        def valid(commit, lastest, oldest):
            return (
                oldest.to_pydatetime().timestamp()
                <= commit.commit_time
                <= lastest.to_pydatetime().timestamp()
            )

        commits = [
            {
                "message": commit.message,
                "sha": commit.hex,
                "author": commit.author,
                "commit_time": commit.commit_time,
                "committer": commit.committer,
            }
            for commit in commits
            if valid(commit, lastest, oldest)
        ]

        return commits

    def clone_repos(self) -> None:
        """Clone github repository"""

        path = os.path.join(REPOS_STORE, self.folder)
        pygit2.clone_repository(
            f"https://github.com/{self.owner}/{self.repo}",
            path,
            callbacks=MyRemoteCallbacks(),
        )
