import os
from sentence_transformers import SentenceTransformer
github_token = "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"
HEADERS = {
    "Authorization": f"token {github_token}",   
    "Accept": "application/vnd.github.v3+json"
}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_THRESHOLD = 0.632
MODEL = SentenceTransformer('all-mpnet-base-v2')
INPUT_VECTOR_LEN = 15
EMBEDED_VECTOR_LEN = 200