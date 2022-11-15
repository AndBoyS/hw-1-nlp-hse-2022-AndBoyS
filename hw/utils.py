from typing import *
from collections import defaultdict
import re
import nltk

def zero_func():
    return 0


def is_stopword(token: str, stopwords: List[str]):
    """
    Проверка на стопслово (список stopwords + еще пару проверок)
    """
    token = token.lower()
    token = re.sub('-', '', token)
    token = re.sub("'", '', token)

    try:
        assert len(token) > 1
        assert token not in ["nt", "re", "ll", "ve"]

        if stopwords:
            assert token not in stopwords

    except AssertionError:
        return True

    return False


def get_word_counters(texts: List[str],
                      stopwords: Optional[Tuple[str]] = None):

    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words('english')

    counter_dict = defaultdict(zero_func)

    for text in texts:

        for token in nltk.word_tokenize(text):

            if is_stopword(token, stopwords):
                continue
            counter_dict[token] = counter_dict.get(token, 0) + 1

    return counter_dict