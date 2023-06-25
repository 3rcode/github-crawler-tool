# This file use data in test1_origin train repos to test whether input commit is important or not
import numpy as np
import os
from base_functions import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
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
    X_train, y_train = load_data(test1_train_repos)
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(X_train)
    
    

def test_encode_cosine(commit):
    pass

def test_nn(commit):
    pass


if __name__ == '__main__':
    pass