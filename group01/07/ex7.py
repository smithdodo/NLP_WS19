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

    with open(corpus) as f_corpus:
        with open(taggings) as f_pos:
            seen = deque(['<S>'], maxlen=2)

            cnt = 0

            while True:
                try:
                    sentance = next(f_corpus)
                    tagging = next(f_pos)
                except StopIteration:
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
                if cnt >= 10000:
                    break
    

    # add-1 smoothing for p(g_i|g_{i-1})
    for key in product(tags, repeat=2):
        bigram_pos[key] += 1

    # add-1 smoothing for p(w|g)
    for w in vocabulary:
        # if w == 'Vinken':
        #     import ipdb; ipdb.set_trace()
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


    print(bigram_pos.head())
    print(p_w_given_g.head())
    return bigram_pos, p_w_given_g, list(tags), list(vocabulary)


def _break_line(line):
    line = line.strip()
    return [el.strip() for el in line.split(' ')]



def tag(sentance, bigram_pos, p_w_given_g, tags, vocabulary):
    words = _break_line(sentance)

    results = []

    print(tags)

    # stores all O(n, g)
    probs = []
    probs.append(
        {g: {'prob':p_w_given_g.loc[words[0], g]['count'], 'g_prev': None} for g in tags})

    for w in words[1:]:
        result = {}
        if w not in vocabulary:
            continue

        for g in tags:

            # {O(n-1, g')*g(g|g') | g'}
            # try:
            prev_probs = np.array([
                probs[-1][g_prev]['prob'] * bigram_pos.loc[g_prev, g]['count'] for g_prev in tags])
            # except KeyError as e:
            #     import ipdb; ipdb.set_trace()
            #     print(e)

            choice = np.argmax(prev_probs)
            prob = p_w_given_g.loc[w, g] * prev_probs[choice]

            result[g] = {'prob': prob, 'g_prev': tags[choice]}


        probs.append(result)


    tagging = []

    idx = np.argmax(np.array([d['prob'] for d in probs[-1].values()]))
    tagging.append(tags[idx])
    last_choice = probs[-1][tagging[0]]['g_prev']
    tagging.insert(0, last_choice)

    for prob in reversed(probs[1:-1]):
        last_choice = prob[last_choice]['g_prev']
        tagging.insert(0, last_choice)
    
    tagging = list(reversed(tagging))
    for w in words:
        if re.match(r'\W+', w): 
            print(w)
            continue
        print(f'{w:10s} {tagging.pop()}')
    # print(tagging)
    # print(words)


if __name__ == "__main__":
    corpus = 'wsj/wsj.text.train'
    pos = 'wsj/wsj.pos.train'
    bigram, p_w_given_g, tags, vocabulary = train(corpus, pos)

    tag(
        "I need a flight from a",
        bigram, p_w_given_g, tags, vocabulary
    )
