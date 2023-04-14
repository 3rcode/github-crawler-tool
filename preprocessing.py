import re
import spacy
import nltk
import numpy as np
import spacy.cli
spacy.cli.download("en_core_web_lg")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

nlp = spacy.load('en_core_web_lg')
bleu_scorer = nltk.translate.bleu_score.sentence_bleu

def preprocess(corpus):
    """Preprocess a corpus.

    Parameters
    ----------
    corpus : str
        Corpus to preprocess.

    Returns
    -------
    str
        Preprocessed corpus.

    """
    global nlp, stopwords
    corpus = corpus.lower()
    # Format words and remove unwanted characters
    corpus = re.sub(r"[^a-zA-Z0-9]", " ", corpus)
    # Tokenize words
    corpus = nlp(corpus)
    # Lemmatize verbs by specifying pos
    corpus = [token.lemma_ if token.pos_ ==
              "VERB" else token.text for token in corpus]
    # Remove spaces
    corpus = [token for token in corpus if not re.match(r"\s+", token)]
    return corpus


def get_scores(sentence1, sentence2):
    """Get scores of two sentences.

    Parameters
    ----------
    sentence1 : str
        Sentence 1.
    sentence2 : str
        Sentence 2.

    Returns
    -------
    int
        Score of two sentences.

    """
    global bleu_scorer
    # Get scores
    score = bleu_scorer([sentence1], sentence2,
                        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
    return score


def score_by_bleu(commit, changelog):
    """Scoring a commit message.

    Parameters
    ----------
    commit : str
        Commit message.
    changelog : str
        Changelog (Release note).

    Returns
    -------
    int
        Score of the commit message.

    """
    # Preprocess commit message
    commit = preprocess(commit)
    commit = " ".join(commit)
    # Preprocess changelog
    changelog_sentences = list(set(changelog.split('\n')))
    changelog_sentences = list(
        map(lambda x: ' '.join(preprocess(x)), changelog_sentences))
    # Get scores
    scores = list(map(lambda sentence: (sentence, get_scores(commit, sentence)), changelog_sentences))
    # Return the maximum score
    return max(scores, key=lambda x: x[1])


# def score_without_break_changelog(commit, changelog):
#     """Scoring a commit message.

#     Parameters
#     ----------
#     commit : str
#         Commit message.
#     changelog : str
#         Changelog (Release note).

#     Returns
#     -------
#     int
#         Score of the commit message.

#     """
#     # Preprocess commit message
#     commit = preprocess(commit)
#     commit = " ".join(commit)
#     # Preprocess changelog
#     changelog = preprocess(changelog)
#     # Get scores
#     scores = get_scores(commit, changelog)
#     return scores


# def score_by_pretrained_model(commit, changelog):
#     global tokenizer, model
#     # Preprocess commit message
#     commit = preprocess(commit)
#     commit = " ".join(commit)
#     # Preprocess changelog
#     changelog_sentences = list(set(changelog.split('\n')))
#     changelog_sentences = list(
#         map(lambda x: ' '.join(preprocess(x)), changelog_sentences))
#     # Get scores
#     max_score = 0
#     sentence = None
#     for changelog_sentence in changelog_sentences:
#         tokens = tokenizer.encode_plus(
#             commit, changelog_sentence, return_tensors='pt')
#         classification_logits = model(**tokens)[0]
#         results = torch.softmax(classification_logits, dim=1).tolist()[0]
#         if results[1] > max_score:
#             max_score = results[1]
#             sentence = changelog_sentence
#     return (sentence, max_score)
