from typing import *
import nltk
from collections import defaultdict
from tqdm import tqdm


def zero_func():
    return 0


def get_word_counters(texts: List[str],
                      stopwords: Optional[Tuple[str]] = None):

    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words('english')

    counter_dict = defaultdict(zero_func)

    for text in tqdm(texts):

        for token in text.split():
            if stopwords and token not in stopwords:
                continue
            counter_dict[token] = counter_dict.get(token, 0) + 1

    return counter_dict