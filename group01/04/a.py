import pandas as pd
import numpy as np
from collections import defaultdict, Counter


def add_count(model, sequence, n=1):
    # N(h, w) += 1
    word = sequence[0]
    if len(sequence) > 1:
        if word not in model:
            model[word] = defaultdict(int) 
        add_count(model[word], sequence[1:], n)
    else:
        model[word] += n


def get_sub_tree(model, prefix):
    if len(prefix) == 0:
        return model

    word = prefix[0]
    if len(prefix) == 0:
        if word not in model:
            return defaultdict(int)
        return model[word]
    else:
        return get_sub_tree(model, prefix[1:])


def get_count(model, sequence):
    # N(h, *)
    if len(sequence) == 0:
        return leaf_sum(model)

    word = sequence[0]
    if len(sequence) > 1:
        try:
            return get_count(model[word], sequence[1:])
        except TypeError:
            # return 0 in case if leaf is reached and sequence still contains word
            return 0
    elif word not in model:
        return 0
    else:
        return model[word]


def leaf_sum(model):
    # sum all leaf values using dfs
    # N(*, *)
    children = list(model.values())
    if len(children) == 0:
        return 0
    elif  isinstance(children[0], int):
        return children[0]
    return sum([leaf_sum(c) for c in children]) 


def count_of_counts(model):
    result = Counter()

    children = list(model.values())
    if isinstance(children[0], int):
        return Counter(children)

    # bfs
    for child in children:
        result += count_of_counts(child)
    
    return result


def train_ngram_model(n=2):
    model = {}
    vocabulary = set()
    with open('Europal-v9', 'r') as f:
        cnt = 0
        for line in f.readlines():
            words = [word for word in line.strip().split(' ') if len(word) > 0]
            vocabulary.update(words)
            for i in range(len(words)):
                if i < n - 1:
                    continue
                sequence = words[i-(n-1):i+1]
                # count corresponding n-gram word sequences
                add_count(model, sequence)
            cnt += 1
            if cnt >= 1000:
                break
    return model, list(vocabulary)


# ================== abs discounting =============
def get_params(model, h, voc):
    b_h = 0.99
    sum_beta_hw = 0

    coc_h = count_of_counts(get_sub_tree(model, h))
    N_h = sum(coc_h.values())

    for v in voc:
        sum_beta_hw += leaf_sum(get_sub_tree(model, h + [v]))
    
    return b_h, sum_beta_hw, coc_h


def absolute_discounting(model, h, w, coc_h, b_h, sum_beta_hw):
    print(w)
    N_hw = get_count(model, h + [w])
    N_h = sum(coc_h.values())

    b_h = 0.95
    W = 1e3 # fix

    if N_hw > 1:
        return (N_hw - b_h) / (N_h)

    beta_hw = N_hw / N_h
    return b_h * ((N_h - coc_h[0]) / N_h) * (beta_hw / sum_beta_hw)


def interpolated_absolute_discounting(model, h, w, b_h, sum_beta_hw):
    N_hw = get_count(model, h + [w])
    coc_h = count_of_counts(get_sub_tree(model, h))
    N_h = sum(coc_h.values())

    b_h = 0.95

    return max((N_hw - b_h), 0) / (N_h) + b_h * (1 - coc_h[0]/N_h) * (N_hw / N_h)

 # =========================================================


def get_sentence(model, voc, length=20):
    coc = count_of_counts(model)

    # using 2-gram

    start = 'I'
    result = [start]
    h = [start]

    for _ in range(length):
        b_h, sum_beta_hw, coc_h = get_params(model, h, voc)
        p = [(w, absolute_discounting(model, h, w, coc_h, b_h, sum_beta_hw)) for w in voc]
        p.sort(key=lambda x: x[1])
        next_word = p[-1][0]
        print(next_word)
        result.append(next_word)

        h.pop(0)
        h.append(next_word)
    
    print(' '.join(result))
    return result
        


def ex5_a():
    # a)
    model, voc = train_ngram_model(n=2)
    # b)
    # coc = count_of_counts(model)

    get_sentence(model, voc, 10)


if __name__ == "__main__":
    ex5_a()
