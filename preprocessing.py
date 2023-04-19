import re
import nltk


bleu_scorer = nltk.translate.bleu_score.sentence_bleu
tokenize = nltk.tokenize.word_tokenize
stemmer = nltk.stem.PorterStemmer()
method = nltk.translate.bleu_score.SmoothingFunction().method1

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
    # Lowercase the corpus
    corpus = corpus.lower()
    # Format words and remove unwanted characters
    corpus = re.sub(r"[^a-zA-Z0-9]", " ", corpus)
    corpus = re.sub(r"\s+", " ", corpus)
    corpus = tokenize(corpus)
    corpus = [stemmer.stem(word) for word in corpus]
    return corpus


def score_by_bleu(commits, changelog):
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
    changelog_sentences = list(set(changelog.split('\n')))
    print('Start to preprocess changelog sentences')
    changelog_sentences = list(map(lambda sentence: preprocess(sentence), changelog_sentences))
    print('Start to preprocess commit messages')
    commits = list(map(lambda commit: preprocess(commit), commits))
    def score_commit(index, commit):
        print(f'Processing commit {index + 1}')
        return bleu_scorer(changelog_sentences, commit, smoothing_function=method)
    scores = list(map(lambda index, commit: score_commit(index, commit), range(len(commits)), commits))
    return scores

