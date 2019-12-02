from collections import deque, defaultdict
from itertools import product
import re

import pandas as pd
import numpy as np


def train(corpus, taggings):
    print('Training model...')
    bigram_pos = defaultdict(int)  # p(g_i|g_{i-1})
    p_w_given_g = defaultdict(int)  # p(w|g)
    vocabulary = set()
    tags = set()

    f = read_files([corpus, taggings])
    seen = deque(['<S>'], maxlen=2)

    cnt = 0

    while True:
        try:
            sentance, tagging = next(f)
        except Exception:
            break

        for word, tag in zip(_break_line(sentance), _break_line(tagging)):
            if not re.match(r'\w+', tag):
                continue

            seen.append(tag)

            # update models
            bigram_pos[tuple(seen)] += 1
            p_w_given_g[(word, tag)] += 1
            vocabulary.add(word)
            tags.add(tag)

        cnt += 1
        # if cnt >= 1000:
        #     break
    
    # add-1 smoothing for p(g_i|g_{i-1})
    for key in product(tags, repeat=2):
        bigram_pos[key] += 1

    # add-1 smoothing for p(w|g)
    for w in vocabulary:
        for tag in tags:
            p_w_given_g[(w, tag)] += 1


    bigram_pos = pd.DataFrame(
        [[*k, v] for k, v in bigram_pos.items()], 
        columns=['g_prev', 'g', 'count'],
    )
    bigram_pos.set_index(['g_prev', 'g'], inplace=True)
    bigram_pos = bigram_pos / bigram_pos.groupby('g_prev').sum()

    p_w_given_g = pd.DataFrame(
        [[*k, v] for k, v in p_w_given_g.items()],
        columns=['w', 'g', 'count'],
    )
    p_w_given_g.set_index(['w', 'g'], inplace=True)
    p_w_given_g = p_w_given_g / p_w_given_g.groupby('g').sum()


    bigram_pos = {k: v['count'] for k, v in bigram_pos.to_dict('index').items()}
    p_w_given_g = {k: v['count'] for k, v in p_w_given_g.to_dict('index').items()}
    return bigram_pos, p_w_given_g, list(tags), list(vocabulary)


def _break_line(line):
    line = line.strip()
    return [el.strip() for el in line.split(' ')]



def tag(sentance, bigram_pos, p_w_given_g, tags, vocabulary):
    words = _break_line(sentance)
    results = []

    # stores all O(n, g)
    probs = []
    if words[0] in vocabulary:
        probs.append(
            {g: {'prob': p_w_given_g[(words[0], g)], 'g_prev': None} for g in tags})
    else:
        probs.append(
            {g: {'prob': 1/len(vocabulary), 'g_prev': None} for g in tags})

    for w in [w for w in words[1:] if is_token(w)]:
        result = {}

        for g in tags:
            # {O(n-1, g')*g(g|g') | g'}
            prev_probs = np.array([
                probs[-1][g_prev]['prob'] * bigram_pos[(g_prev, g)] for g_prev in tags])
            choice = np.argmax(prev_probs)

            try:
                member_ship_prob = p_w_given_g[(w, g)]
            except KeyError:
                # handling OOV
                member_ship_prob = 1 / len(vocabulary)

            prob = member_ship_prob * prev_probs[choice]
            result[g] = {'prob': prob, 'g_prev': tags[choice]}

        probs.append(result)

    # select best tagging
    tagging = []

    idx = np.argmax(np.array([d['prob'] for d in probs[-1].values()]))
    tagging.append(tags[idx])
    last_choice = probs[-1][tagging[0]]['g_prev']
    tagging.append(last_choice)

    for prob in reversed(probs[1:-1]):
        last_choice = prob[last_choice]['g_prev']
        tagging.append(last_choice)
    
    tagging.reverse()

    result = []
    for i, w in enumerate(words):
        if not is_token(w):
            # print(w)
            result.append(w)
        else:
            result.append(tagging.pop(0))

        print(f'{w:10s} -> {result[-1]}')

    return result


def is_token(w):
    return re.match(r'^\W+$', w) is None


def read_files(paths):
    files = [open(path, 'r') for path in paths]
    try:
        while True:
            yield [next(f) for f in files]
    except StopIteration:
        for f in files:
            f.close()
        raise


def score(bigram, p_w_given_g, tags, vocabulary):
    corpus = 'wsj/wsj.text.test'
    pos = 'wsj/wsj.pos.test'
    f = read_files([corpus, pos])

    result = []

    cnt = 0
    while True:

        try:
            X, y = next(f)
            y = y.strip().split(' ')
            tagging = tag(X, bigram, p_w_given_g, tags, vocabulary)
            result += list(zip(y, tagging))
        except Exception:
            break

        cnt += 1
        # if cnt > 10:
        #     break
    
    print(
        'Accuracy on test data:',
         sum([1 if (y[0] == y[1] or re.match('^\W+$', y[0])) else 0 for y in result])/len(result)
    )


if __name__ == "__main__":
    corpus = 'wsj/wsj.text.train'
    pos = 'wsj/wsj.pos.train'
    bigram, p_w_given_g, tags, vocabulary = train(corpus, pos)
    
    score(bigram, p_w_given_g, tags, vocabulary)

    # tag(
    #     "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 . ",
    #     bigram, p_w_given_g, tags, vocabulary
    # )
