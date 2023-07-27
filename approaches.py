# import os
# from keras.models import Sequential, load_model
# from keras.layers import Dense, LSTM, Embedding, TextVectorization
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from settings import MODEL, ROOT_DIR, INPUT_VECTOR_LEN, EMBEDDED_VECTOR_LEN
# from sklearn.metrics.pairwise import cosine_similarity
# from base_functions import load_data, analyze_result, show_result, save_result


# def LSTM_model(test_name, _type, X_train, y_train, X_test, y_test):
#     # Build model
#     vectorize_layer = TextVectorization(
#         split="whitespace",
#         output_mode="int",
#         pad_to_max_tokens=True,
#         output_sequence_length=INPUT_VECTOR_LEN           
#     )
#     vectorize_layer.adapt(X_train)
#     top_words = len(vectorize_layer.get_vocabulary()) + 1
#     model = Sequential()
#     model.add(vectorize_layer)
#     model.add(Embedding(top_words, EMBEDDED_VECTOR_LEN, input_length=INPUT_VECTOR_LEN))
#     model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, unroll=True))
#     model.add(Dense(1, activation="sigmoid"))
#     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "precision", "recall", "auc"])
    
#     # Load model has trained in previous running session
#     # model_file = os.path.join(ROOT_DIR, "models", "lstm_models", f"{test_name}_{_type}"")
#     # model = load_model(model_file)

#     # Check build_model function
#     print(model.summary())
    
#     # Train model
#     model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1) 
    
#     # Save model 
#     model_file = os.path.join(ROOT_DIR, "models", "lstm_models", f"{test_name}_{_type}")
#     model.save(model_file)  

#     # Test model
#     y_preds = model.predict(X_test)
#     y_preds = [1 if x > 0.5 else 0 for x in y_preds]
#     path = os.path.join(ROOT_DIR, "sample_wrong_cases", "LSTM_model.yml")
#     result = analyze_result(path, (test_name, _type), X_test, y_preds, y_test)
#     show_result(result)
#     result_path = os.path.join(ROOT_DIR, "LSTM_model.yml")
#     save_result(result_path, (test_name, _type), result)

        
# def naive_bayes(test_name, _type, X_train, y_train, X_test, y_test, vectorizer):
#     X_train = vectorizer.fit_transform(X_train)
#     print(f"\nNum Samples: {X_train.shape[0]}\nNum Features: {X_train.shape[1]}")
#     model = MultinomialNB()
#     # Train model
#     model.fit(X_train, y_train)
#     # Test
#     test_commits = X_test
#     X_test = vectorizer.transform(X_test)
#     y_preds = model.predict(X_test)
#     path = os.path.join(ROOT_DIR, "sample_wrong_cases", "naive_bayes.yml")
#     result = analyze_result(path, (test_name, _type), test_commits, y_preds, y_test)
#     show_result(result)
#     result_path = os.path.join(ROOT_DIR, "naive_bayes.yml")
#     save_result(result_path, (test_name, _type), result)
    

# def encode_cosine(test_name, _type, X_train, y_train, X_test, y_test):
#     file_name = 'labeled_commits.csv' if _type == 'origin' else 'labeled_commits_abstract.csv'
#     # load changelog sentences database
#     all_changelog_sentences = []

#     for repo in train_repos:
#         path = os.path.join(data_path, repo, 'release_notes.csv')
#         df = pd.read_csv(path)
#         for release_note in df['Release Note']:
#             changelog_sentences = str(release_note).split('\n')
#             all_changelog_sentences.extend(changelog_sentences)

#     # Remove duplicate sentences and linking statements
#     all_changelog_sentences = list(set(all_changelog_sentences))
#     all_changelog_sentences = [sentence for sentence in all_changelog_sentences  
#                                     if not (any(re.match(pattern, sentence) for pattern in linking_statement) or sentence == '')]
    
#     X_test = []
#     y_test = []
#     # Load commit messages and expected label of commits
#     for repo in test_repos:
#         path = os.path.join(data_path, repo, file_name)
#         commit, label = load_data(path)
#         X_test = np.concatenate((X_test, commit), axis=0)
#         y_test = np.concatenate((y_test, label), axis=0)

#     # Encode all changelog sentences
#     print('Start to encode changelog sentences')
#     # Load encoded changelog sentences
#     # encoded_file = os.path.join(ROOT_DIR, 'models', 'encoded_vectors', f'{test_name}_{_type}.npy')
#     # with open(encoded_file, 'rb') as f:
#     #     encoded_changelog_sentences = np.load(f)

#     # Encode and save encoded changelog sentences
#     encoded_changelog_sentences = model.encode(all_changelog_sentences)
#     print('Successfully encoded changelog sentences')
#     print('Encoded changelog sentences shape:', encoded_changelog_sentences.shape)
#     encoded_file = os.path.join(ROOT_DIR, 'models', 'encoded_vectors', f'{test_name}_{_type}.npy')
#     with open(encoded_file, 'wb') as f:
#         np.save(f, encoded_changelog_sentences)

#     # Encode test commit
#     print('Start to encode test commit')
#     encoded_test_commit = model.encode(X_test)
#     print('Successfully encoded test commit')
#     print('Encoded test commit shape:', encoded_test_commit.shape)

#     # print('Start to calculate cosine similarity')
#     scores = np.asarray([])
#     # Split commit into some batchs (can't work with whole commits because of numpy max allocate memory capacity)
#     rounds = len(encoded_test_commit) // BATCH_SIZE
#     for i in range(rounds):
#         cosine_similarities = cosine_similarity(encoded_test_commit[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], encoded_changelog_sentences)
#         scores = np.concatenate((scores, np.amax(cosine_similarities, axis=1)), axis=0)
#     # Calculate rest commits
#     cosine_similarities = cosine_similarity(encoded_test_commit[rounds * BATCH_SIZE:], encoded_changelog_sentences)
    
    
#     scores = np.concatenate((scores, np.amax(cosine_similarities, axis=1)), axis=0)
#     # Save scores of each test so that no need to wait a long time to see result next time
#     score_file = os.path.join(ROOT_DIR, 'models', 'approach2_scores', f'{test_name}_{_type}.npy')
#     with open(score_file, 'wb') as f:
#         np.save(f, scores)
    
#     # Load scores from previous running session
#     # score_file = os.path.join(ROOT_DIR, 'models', 'approach2_scores', f'{test_name}_{_type}.npy')
#     # with open(score_file, 'rb') as f:
#     #     scores = np.load(f)
#     y_preds = np.where(scores >= THRESHOLD, 1, 0)
#     print('Successfully calculated cosine similarity')
#     # Score result
#     path = os.path.join(ROOT_DIR, 'sample_wrong_cases', 'encode_cosine.yaml')
#     result = analyze_result(path, (test_name, _type), X_test, y_preds, y_test)
#     show_result(result)
#     result_path = os.path.join(ROOT_DIR, 'encode_cosine.yaml')
#     save_result(result_path, (test_name, _type), result)


# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     df = df.dropna(subset=["Commit Message"]).reset_index(drop=True)
#     def merge(row):
#         if pd.isna(row["Commit Description"]):
#             return row["Commit Message"]
#         return row["Commit Message"] + "<.> " + row["Commit Description"]
    
#     commit = df.apply(merge, axis=1)
#     label = df["Label"]
#     return np.asarray(commit), np.asarray(label)


# def load_data(test_name, _type="origin", adjust_train_data=False, over_sampling=1, under_sampling=1):
#     path = os.path.join(ROOT_DIR, f"datasets_{_type}", test_name)
#     train_data = pd.read_csv(os.path.join(path, 'train.csv'))
#     test_data = pd.read_csv(os.path.join(path, 'test.csv'))
    
#     if adjust_train_data:
#         train_data_label_1 = train_data[train_data["y_train"] == 1]
#         train_data_label_0 = train_data[train_data["y_train"] == 0]
#         train_data_label_1 = train_data_label_1.sample(frac=over_sampling, replace=True, random_state=0)
#         train_data_label_0 = train_data_label_0.sample(frac=under_sampling, replace=False, random_state=1)
#         train_data = pd.concat([train_data_label_0, train_data_label_1])
#         train_data = train_data.sample(frac=1, random_state=2)

#         X_train = train_data["X_train"]
#         y_train = train_data["y_train"]
#         X_test = test_data["X_test"]
#         y_test = test_data["y_test"]
#     return X_train, y_train, X_test, y_test


# def save_result(result_path, test_case, result):
#     test_name, _type = test_case
#     total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn = result
#     # Need to save result into file
#     _save_result(result_path, {f"{test_name}_{_type}": {"Total test": total_test, 
#                                                         "Precision": precision,
#                                                         "Recall": recall,
#                                                         "F1 score": f1_score,
#                                                         "Accuracy": accuracy,
#                                                         "True negative rate": true_neg_rate,
#                                                         "True positive": tp,
#                                                         "True negative": tn,
#                                                         "False positive": fp,
#                                                         "False negative": fn
#                                                         }})


# def _save_result(path, content):
#     with open(path, 'r') as f:
#         result = yaml.safe_load(f)
#         if result is None:
#             result = {}
#         result.update(content)
#     with open(path, 'w') as f:
#         yaml.safe_dump(result, f)


# def analyze_result(path, test_case, commits, prediction, Y):
#     test_name, _type = test_case
#     tp = sum([pred == y and pred == 1 for pred, y in zip(prediction, Y)])
#     tn = sum([pred == y and pred == 0 for pred, y in zip(prediction, Y)])
#     fp = sum([pred != y and pred == 1 for pred, y in zip(prediction, Y)])
#     fn = sum([pred != y and pred == 0 for pred, y in zip(prediction, Y)])

#     # false_case = [index for index in range(len(Y)) if prediction[index] != Y[index]]
#     # true_case = [index for index in range(len(Y)) if prediction[index] == Y[index]]
#     # true_positive = [commits[index] for _, index in enumerate(true_case) if prediction[index] == 1]
#     # true_negative = [commits[index] for _, index in enumerate(true_case) if prediction[index] == 0]
#     # false_positive = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 1]
#     # false_negative = [commits[index] for _, index in enumerate(false_case) if prediction[index] == 0]
#     # _save_result(path, {f'{test_name}_{_type}': {'False Positive': false_positive, 
#     #                                              'False Negative': false_negative}})
#     # tp = len(true_positive)
#     # tn = len(true_negative)
#     # fp = len(false_positive)
#     # fn = len(false_negative)
    
#     total_test = len(Y)
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     true_neg_rate = tn / (tn + fp)
#     f1_score = 2 * precision * recall / (precision + recall)
#     accuracy = (tp + tn) / (tp + tn + fp + fn)

    
#     return total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn


# def show_result(result):
#     total_test, precision, recall, f1_score, accuracy, true_neg_rate, tp, tn, fp, fn = result
#     print("Total test:", total_test)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 score:", f1_score)
#     print("Accuracy:", accuracy)
#     print("True negative rate:", true_neg_rate)
#     print("True positive:", tp)
#     print("True negative:", tn)
#     print("False positive:", fp)
#     print("False negative", fn)
