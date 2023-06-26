# This file use data in test1_origin train repos to test whether input commit is important or not
import numpy as np
import os
from base_functions import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from base_functions import find_commit
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keras.models import load_model


model = SentenceTransformer('all-mpnet-base-v2')
THRESHOLD = 0.7
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
repos = []
data_path = os.path.join(ROOT_DIR, 'data')
for subdir, dirs, files in os.walk(data_path):
    repos.extend(dirs)
test1_test_repos = ['keystonejs_keystone',
                    'hashicorp_terraform-cdk',
                    'jenkinsci_jenkins',
                    'vercel_swr',
                    'hashicorp_terraform-provider-azurerm',
                    'elementary_calculator',
                    'nestjs_bull',
                    'irccloud_android',
                    'tailwindlabs_tailwindcss',
                    'hashicorp_terraform']

def test_naive_bayes(commit):   
    test1_train_repos = list(set(repos) - set(test1_test_repos))
    X_train = np.asarray([])
    y_train = np.asarray([])
    for repo in test1_test_repos:
        path = os.path.join(data_path, repo, 'labeled_commits.csv')
        cm, label = load_data(path)
        X_train = np.concatenate((X_train, cm), axis=0)
        y_train = np.concatenate((y_train, label), axis=0)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    vectorized_commit = vectorizer.transform([commit])
    prediction = model.predict(vectorized_commit)
    print("Predict commit is in class:", prediction[0])
    commit_info = find_commit(commit)
    print("Commit info:")
    print(commit_info)
    

def test_encode_cosine(commit):
    path_test1_origin = os.path.join(ROOT_DIR, 'models', 'encoded_vectors', 'test1_origin.npy')
    print(path_test1_origin)
    with open(path_test1_origin, 'rb') as f:
        encoded_changelog_sentences = np.load(f)
    encoded_commit = model.encode([commit])
    scores = cosine_similarity(encoded_commit, encoded_changelog_sentences)[0]
    max_score = max(scores)
    print("Commit's score is:", max_score)  
    commit_info = find_commit(commit)
    print("Commit info:")
    print(commit_info)


def test_nn(commit):
    model_path = os.path.join(ROOT_DIR, 'models', 'lstm_models', 'test1_origin')
    model = load_model(model_path)
    print(model.summary())
    print("Commit's score is:", model.predict([commit])[0])
    commit_info = find_commit(commit)
    print("Commit info:")
    print(commit_info)


if __name__ == '__main__':
    # Test 3 approachs here
    # Input is commit want to test
    test_nn("Hello world")
    test_naive_bayes("Hello world")
    test_encode_cosine("Hello world")