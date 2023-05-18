import os
import pandas as pd
import numpy as np
# import random
import re
from main import load_data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

linking_statement = [r"(?i)fixes:?", r"(?i)what's changed:?", r"(?i)other changes:?", r"(?i)documentation:?",
                     r"(?i)features:?", r"(?i)new contributors:?", r"(?i)bug fixes:?", r"(?i)changelog:?",
                     r"(?i)notes:?", r"(?i)improvements:?", r"(?i)deprecations:?", r"(?i)minor updates:?",
                     r"(?i)changes:?", r"(?i)fixed:?", r"(?i)maintenance:?", r"(?i)added:?"]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(ROOT_DIR, 'data')
repos = []
for subdir, dirs, files in os.walk(data_path):
    repos.extend(dirs)
test_repo = 'vercel_swr' # random.choice(repos)
repos.remove(test_repo)

# load changelog sentences database
all_changelog_sentences = []
for repo in repos:
    path = os.path.join(data_path, repo, 'release_notes.csv')
    df = pd.read_csv(path)
    for release_note in df['Release Note']:
        changelog_sentences = str(release_note).split('\n')
        all_changelog_sentences.extend(changelog_sentences)

# Remove duplicate sentences and linking statements
all_changelog_sentences = list(set(all_changelog_sentences))
all_changelog_sentences = [sentence for sentence in all_changelog_sentences  
                                   if not (any(re.match(pattern, sentence) for pattern in linking_statement) or sentence == '')]
# Load test data
test_path = os.path.join(data_path, test_repo, 'labeled_commits.csv')
test_commit, test_label = load_data(test_path)

# Encode all changelog sentences
print('Start to encode changelog sentences')
# with open('models/encoded_changelog_sentences.npy', 'rb') as f:
#     encoded_changelog_sentences = np.load(f)
encoded_changelog_sentences = model.encode(all_changelog_sentences)
print('Successfully encoded changelog sentences')
print('Encoded changelog sentences shape:', encoded_changelog_sentences.shape)
with open ('models/encoded_changelog_sentences.npy', 'wb') as f:
    np.save(f, encoded_changelog_sentences)

# Encode test commit
print('Start to encode test commit')
encoded_test_commit = model.encode(test_commit)
print('Successfully encoded test commit')
print('Encoded test commit shape:', encoded_test_commit.shape)

# Calculate cosine similarity
print('Start to calculate cosine similarity')
cosine_similarities = cosine_similarity(encoded_test_commit, encoded_changelog_sentences)
pred = [0] * len(cosine_similarities)
for i, score in enumerate(cosine_similarities):
    result = max(score)
    if result > 0.7: 
        pred[i] = 1
    else:
        pred[i] = 0

print('Successfully calculated cosine similarity')

# Score result
test_size = len(test_commit)
true_pred = sum([pred[i] == test_label[i] for i in range(test_size)])
print("Accuracy: %.2f%%" % (true_pred / test_size * 100))
score = f1_score(test_label, pred)
print(f"F1 score: {score}")

