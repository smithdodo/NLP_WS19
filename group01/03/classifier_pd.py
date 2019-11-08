import pandas as pd
import numpy as np
from collections import Counter


def get_voc(size, path):
    with open(path, 'r') as f:
        all_vocs = [w.strip() for w in f.read().split('\n')]
        limit = size if size < len(all_vocs) else len(all_vocs)
        voc = {v.strip(): 0 for v in all_vocs[:limit]}
        voc['<UNK>'] = 0
    return voc


def read_training_data(text_path, voc):
    with open(text_path, 'r') as f:
        rows = []
        for line in f.readlines():
            row = voc.copy()
            elements = [el for el in line.strip().split(' ') if el]
            row['class_name'] = elements[1]
            for word, count in zip(elements[2::2], elements[3::2]):
                if word in row:
                    row[word] += int(count)
                else:
                    row['<UNK>'] += int(count)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('class_name')
    return df


def train(df):
    """
    Return log likelyhood p(w|c) and prior p(c)
    """
    # log p(c)
    pc = np.log10(pd.DataFrame([{k: v/len(df) for k, v in Counter(df.index).items()}]))

    # convert word counts to relative freq
    df = df.groupby('class_name').sum()
    # log p(w|c)
    df = np.log10(df.div(df.values.sum(axis=1), axis=0))

    df.replace([np.inf, -np.inf], np.nan)
    del df['<UNK>']

    return df, pc


def predict(word_counts, p_w_c, prior):
    """
    return argmax_c{log{p(c)} + sum_w{N_w * p(w|c)}}
    """
    res = np.nan_to_num(p_w_c.values * word_counts).sum(axis=1) + prior.values
    best_class = prior.columns[res.argmax()]
    return best_class


def evaluate():
    voc = get_voc(500, '20news/20news.voc')
    # p_w_c, prior = train(read_training_data('20news/20news.tr', voc))
    # test = read_training_data('20news/20news.te', voc)
    p_w_c, prior = train(read_training_data('spam/spam.tr', voc))
    test = read_training_data('spam/spam.te', voc)

    results = []
    for idx, row in test.iterrows():
        actual = row.name
        del row['<UNK>']
        pred = predict(row.values, p_w_c, prior)
        results.append((actual, pred))
        # print(f'actual: {actual}, pred: {pred}')

    print('err:', len([1 for i in results if i[0] != i[1]]) / len(results))


if __name__ == '__main__':
    evaluate()
