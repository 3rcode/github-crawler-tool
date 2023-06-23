import numpy as np
import pandas as pd
import os
import yaml
from yaml.loader import SafeLoader
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, TextVectorization
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from base_functions import load_data, save_result, sample_wrong_cases

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
max_commit_length = 30


if __name__ == '__main__':
    repos = []
    data_path = os.path.join(ROOT_DIR, 'data')
    for subdir, dirs, files in os.walk(data_path):
        repos.extend(dirs)
    
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
            vectorize_layer = TextVectorization(
                standardize='strip_punctuation',
                split="whitespace",
                output_mode="int",
                output_sequence_length=max_commit_length              
            )
            vectorize_layer.adapt(X_train)
            top_words = len(vectorize_layer.get_vocabulary()) + 1
            embedding_vector_length = 300
            # Build model
            model = Sequential()
            model.add(vectorize_layer)
            model.add(Embedding(top_words, embedding_vector_length, input_length=max_commit_length))
            model.add(LSTM(32, dropout=0.05, recurrent_dropout=0.05, unroll=True))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Load model
            # model_file = os.path.join(ROOT_DIR, 'models', 'lstm_models', f'{test_name}_{_type}')
            # model = load_model(model_file)

            # Check build_model function
            print(model.summary())
            
            # Train model
            model.fit(X_train, y_train, epochs=3, batch_size=64) 
            
            # Save model 
            model_file = os.path.join(ROOT_DIR, 'models', 'lstm_models', f'{test_name}_{_type}')
            model.save(model_file)

            # Test model
            test_size = len(X_test)
            accuracy = model.evaluate(X_test, y_test, verbose=0)[1] * 100
            print("Accuracy: %.2f%%" % (accuracy))
            accuracy = str(int(accuracy * 100) / 100) + '%'
            y_preds = model.predict(X_test)
            y_preds = [1 if x > 0.5 else 0 for x in y_preds]
            f1 = f1_score(y_test, y_preds)
            print(f"F1 score: {f1}")
            f1 = str(f1)
            result_path = os.path.join(ROOT_DIR, 'LSTM_model.yaml')
            save_result(result_path, test_case=(test_name, _type), result=(test_size, accuracy, f1))

        def naive_bayes(test_name, _type, X_train, y_train, X_test, y_test):
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            y_train = np.asarray(y_train).astype(np.float32)
            print(f"Num Samples: {X_train.shape[0]}\nNum Features: {X_train.shape[1]}")
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            test_size = len(X_test)
            test_commits = X_test
            X_test = vectorizer.transform(X_test)
            y_test = np.asarray(y_test).astype(np.float32)
            print(f"Num Test: {X_test.shape[0]}")
            y_pred = model.predict(X_test)
            true_pred = sum([y_pred[i] == y_test[i] for i in range(test_size)])
            path = os.path.join(ROOT_DIR, 'sample_wrong_cases', 'naive_bayes.yaml')
            sample_wrong_cases(path, (test_name, _type), test_commits, y_pred, y_test)
            accuracy = true_pred / test_size * 100
            print("Accuracy: %.2f%%" % (accuracy))
            accuracy = str(int(accuracy * 100) / 100) + '%'

            f1 = f1_score(y_test, y_pred)
            print(f"F1 score: {f1}")
            f1 = str(f1)
            result_path = os.path.join(ROOT_DIR, 'naive_bayes.yaml')
            save_result(result_path, test_case=(test_name, _type), result=(test_size, accuracy, f1))
        
        naive_bayes(test_name, _type, X_train, y_train, X_test, y_test) 
    
    for test_case, test_repos in test_cases.items():
        train_repos = list(set(repos) - set(test_repos))
        approach1(test_name=test_case, train_repos=train_repos, test_repos=test_repos, _type='origin') 
        
    