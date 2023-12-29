import os

GITHUB_TOKEN = ""
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOS_STORE = os.path.join(ROOT_DIR, "repos")  # Here is where you store clone repositories
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",   
    "Accept": "application/vnd.github.v3+json"
}
