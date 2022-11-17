from typing import *
from collections import defaultdict
import re
from multiprocessing import cpu_count, Pool
from tqdm.notebook import tqdm
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


def get_word_counters_and_tokenize(
        texts: List[str],
        stopwords: Optional[Tuple[str]] = None
    ):

    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words('english')

    counter_dict = defaultdict(zero_func)
    tokenized_texts = [''] * len(texts)

    for i, text in enumerate(texts):

        tokenized_text = nltk.word_tokenize(text)
        tokenized_text = list(filter(lambda x: not is_stopword(x, stopwords),
                                     tokenized_text))

        tokenized_texts[i] = tokenized_text

        for token in tokenized_text:

            counter_dict[token] = counter_dict.get(token, 0) + 1

    return counter_dict, tokenized_texts


def chunks(l, n):
    """
    Разбить список на чанки длины n
    """
    for i in range(0, n):
        yield l[i::n]


def get_word_counters_mp(texts: List[str], n_jobs=-1):

    if n_jobs == -1:
        n_jobs = cpu_count() - 1

    texts = list(chunks(texts, n_jobs))

    with Pool(n_jobs) as p:
        counters_and_tokenized_texts = p.map(get_word_counters_and_tokenize, texts)

    # Объединяем результаты процессов
    res_counter = defaultdict(lambda: 0)
    tokenized_texts = []

    for counter, _tokenized_texts in counters_and_tokenized_texts:
        for name, val in counter.items():
            res_counter[name] += val
        tokenized_texts.extend(_tokenized_texts)

    return res_counter, tokenized_texts


def create_surname_dict(names):
    name_to_surname = {}
    for name in names:
        name_surname = name.split()
        if len(name_surname) == 1:
            name_surname.append('')
        assert len(name_surname) == 2
        name, surname = name_surname
        name_to_surname[name] = surname
    return name_to_surname


def count_names(tokenized_texts, names):

    name_to_surname = create_surname_dict(names)

    name_counter = defaultdict(zero_func)
    name_pair_counter = defaultdict(zero_func)
    prof_name_counter = defaultdict(zero_func)

    for tokens in tokenized_texts:

        pos_tags = nltk.pos_tag(tokens)
        ne_chunks = nltk.ne_chunk(pos_tags)
        subtrees = ne_chunks.subtrees()

        for subtree in subtrees:
            if subtree.label() != 'PERSON':
                continue

            subtree_len = len(subtree)

            for i, (name, pos) in enumerate(subtree):
                surname = name_to_surname.get(name, None)

                is_name = surname is not None
                if is_name:
                    # На самом деле это не обязательно так
                    # но для задачи нам не нужно различать это
                    is_surname = False
                else:
                    is_surname = name in name_to_surname.values()
                # Если name это имя без фамилии / name это фамилия

                # Чекаем есть ли "professor" в прошлом слове
                if (is_name or is_surname) and i > 0:

                    prev_word = subtree[i-1][0].lower()
                    if 'professor' == prev_word:
                        prof_name = ' '.join([prev_word, name])
                        prof_name_counter[prof_name] += 1

                if not is_name and not is_surname:
                    continue

                # Чекаем след слово на то, является ли оно фамилией
                if surname != '' and i < subtree_len - 1:
                    if subtree[i+1][0] == surname:
                        full_name = ' '.join([name, surname])
                        name_pair_counter[full_name] += 1

                name_counter[name] += 1

    return name_counter, name_pair_counter, prof_name_counter


def count_names_mp(tokenized_texts, names, n_jobs=-1):
    """
    names - список имен из Гарри Поттера, элемент либо состоит из одного имени
    """

    if n_jobs == -1:
        n_jobs = cpu_count()
    tokenized_texts = list(chunks(tokenized_texts, n_jobs))
    pool_inputs = [(t, names) for t in tokenized_texts]

    with Pool(n_jobs) as p:
        pool_output = p.starmap(count_names, pool_inputs)

    name_counters = [l[0] for l in pool_output]
    name_pair_counters = [l[1] for l in pool_output]
    prof_name_counters = [l[2] for l in pool_output]

    # Объединяем ответы процессов
    counters_dict = {
        'name_counter': defaultdict(lambda: 0),
        'name_pair_counter': defaultdict(lambda: 0),
        'prof_name_counter': defaultdict(lambda: 0),
    }

    for counters, (new_counter) in zip([name_counters, name_pair_counters, prof_name_counters],
                                       counters_dict.values()):

        for counter in counters:
            for name, val in counter.items():
                new_counter[name] += val


    return counters_dict['name_counter'], counters_dict['name_pair_counter'], counters_dict['prof_name_counter']