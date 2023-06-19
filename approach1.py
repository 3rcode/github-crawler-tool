import numpy as np
import pandas as pd
import os
import yaml
from yaml.loader import SafeLoader
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
max_commit_length = 30


def load_origin_data(file_path):
    df = pd.read_csv(file_path)
    commit = df['Commit Message'].astype(str) + df['Commit Description'].astype(str)
    label = df['Label']
    return np.asarray(commit), np.asarray(label)

def load_abstract_data(file_path):
    df = pd.read_csv(file_path)
    commit = df['Commit Message Abstract'].astype(str) + df['Commit Desription Abstract'].astype(str) # Note description
    label = df['Label']
    return np.asarray(commit), np.asarray(label)

def create_dictionary(commits, word2idx):
    """Create a dictionary of words from commits.

    Parameters
    ----------
    commits : list
        List of commits.

    Returns
    -------
    dict
        Dictionary of words.

    """
    print("Creating dictionary")
    word_count = {}
    for commit in commits:
        lst_words = commit.split(' ')
        for word in lst_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    for idx, (word, count) in enumerate(sorted_word_count):
        word2idx[word] = idx + 1
    word2idx['unk'] = len(word2idx) + 1

def convert2vector(commit, word2idx):
    """Convert commit to a vector.

    Parameters
    ----------
    commit: str
        Commit.

    Returns
    -------
    list
        Vector of the commit.

    """
    vector = []
    commit = commit.split(' ')
    for word in commit:
        if word in word2idx:
            vector.append(word2idx[word])
        else:
            vector.append(word2idx['unk'])
    if len(vector) >= max_commit_length:
        vector = vector[:max_commit_length]
    else:
        vector = [0] * (max_commit_length - len(vector)) + vector
    return vector

def build_model(topwords, embedding_vector_len, input_length):
    """Build the model.
    
    Returns
    -------
    keras.Sequential
        Model.

    """
    model = Sequential()
    model.add(Embedding(topwords, embedding_vector_len, input_length=input_length))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    repos = []
    data_path = os.path.join(ROOT_DIR, 'data')
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    X = np.asarray([])
    y = np.asarray([]) 
    
    test_cases_path = os.path.join(ROOT_DIR, 'test_cases.yaml')
    test_cases = None
    with open(test_cases_path, 'r') as f:
        test_cases = yaml.load(f, Loader=SafeLoader)
    
    def approach1(test_name, train_repos, test_repos, _type):
        X_train = [] 
        X_test = []
        y_train = []
        y_test = []
        file_name = 'labeled_commits.csv' if _type == 'origin' else 'labeled_commits_abstract.csv' 
        load_data = load_origin_data if _type == 'origin' else load_abstract_data

        for repo in train_repos:
            path = os.path.join(data_path, repo, file_name)
            commit, label = load_data(path)
            X_train = np.concatenate((X_train, commit), axis=0)
            y_train = np.concatenate((y_train, label), axis=0)
        
        # Check number of dataset
        # print(X_train.shape)
        # print(y_train.shape)
        
        # Load commit messages and expected label of commits
        for repo in test_repos:
            path = os.path.join(data_path, repo, file_name)
            commit, label = load_data(path)
            X_test = np.concatenate((X_test, commit), axis=0)
            y_test = np.concatenate((y_test, label), axis=0)
        
        # print(X_test.shape)
        # print(y_test.shape)

        def LSTM_model(test_name, _type, X_train, y_train, X_test, y_test):
            word2idx = {}
            create_dictionary(X_train, word2idx)
            # Check create_dictionary function
            print(f"Length of dictionary: {len(word2idx)}")

            # X_train = np.asarray([convert2vector(commit, word2idx) for commit in X_train])
            # Check convert to vector function
            # print(X_train.shape)

            # top_words = len(word2idx) + 1
            # embedding_vector_length = 300
            # classify_model = build_model(top_words, embedding_vector_length, max_commit_length)
            
            # Load model
            model_file = os.path.join(ROOT_DIR, 'models', 'lstm_models', f'{test_name}_{_type}')
            classify_model = load_model(model_file)

            # Check build_model function
            print(classify_model.summary())
            
            # Train model
            # classify_model.fit(X_train, y_train, epochs=3, batch_size=64) 
            
            # Save model 
            # model_file = os.path.join(ROOT_DIR, 'models', 'lstm_models', f'{test_name}_{_type}')
            # classify_model.save(model_file)

            # Test model
            X_test = np.asarray([convert2vector(commit, word2idx) for commit in X_test])
            test_size = len(X_test)
            accuracy = classify_model.evaluate(X_test, y_test, verbose=0)[1] * 100
            print("Accuracy: %.2f%%" % (accuracy))
            accuracy = str(int(accuracy * 100) / 100) + '%'
            y_preds = classify_model.predict(X_test)

            y_preds = [1 if x > 0.5 else 0 for x in y_preds]
            
            f1 = f1_score(y_test, y_preds)
            print(f"F1 score: {f1}")
            f1 = str(f1)

            result_path = os.path.join(ROOT_DIR, 'LSTM_model.yaml')
            with open(result_path, 'r') as f:
                result = yaml.safe_load(f)
                if result is None:
                    result = {}
                result.update({f'{test_name}_{_type}': {'Num Commits': test_size, 'Accuracy': accuracy, 'F1 score': f1}})
            with open(result_path, 'w') as f:
                yaml.safe_dump(result, f)

        
        def naive_bayes(test_name, _type, X_train, y_train, X_test, y_test):
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            y_train = np.asarray(y_train).astype(np.float32)
            print(f"Num Samples: {X_train.shape[0]}\nNum Features: {X_train.shape[1]}")
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            test_size = len(X_test)
            X_test = vectorizer.transform(X_test)
            y_test = np.asarray(y_test).astype(np.float32)
            print(f"Num Test: {X_test.shape[0]}")
            y_pred = model.predict(X_test)
            true_pred = sum([y_pred[i] == y_test[i] for i in range(test_size)])

            accuracy = true_pred / test_size
            print("Accuracy: %.2f%%" % (accuracy * 100))
            accuracy = str(int(accuracy * 10000) / 100) + '%'

            f1 = f1_score(y_test, y_pred)
            print(f"F1 score: {f1}")
            f1 = str(f1)

            result_path = os.path.join(ROOT_DIR, 'naive_bayes.yaml')
            with open(result_path, 'r') as f:
                result = yaml.safe_load(f)
                if result is None:
                    result = {}
                result.update({f'{test_name}_{_type}': {'Num Commits': test_size, 'Accuracy': accuracy, 'F1 score': f1}})
            with open(result_path, 'w') as f:
                yaml.safe_dump(result, f)
        
        naive_bayes(test_name, _type, X_train, y_train, X_test, y_test) 
    
    for test_case, test_repos in test_cases.items():
        train_repos = list(set(repos) - set(test_repos))
        approach1(test_name=test_case, train_repos=train_repos, test_repos=test_repos, _type='origin') 
    
    
    