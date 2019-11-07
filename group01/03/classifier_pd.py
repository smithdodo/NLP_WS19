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
                    row['<UNK>'] += 1
            rows.append(row)

    df = pd.DataFrame(rows)
    # put class_name on first col
    cols = df.columns.tolist()
    cols.remove('class_name')
    cols.insert(0, 'class_name')
    df = df[cols]
    return df


def train(df):
    """
    Return log likelyhood p(w|c) and prior p(c)
    """
    # p(c)
    pc = pd.DataFrame([{k: v/len(df) for k, v in Counter(df['class_name']).items()}])

    # convert word counts to relative freq
    df['length'] = df.iloc[:, 1:-1].sum(axis=1)
    df = df.groupby('class_name').sum()
    # log p(w|c)
    df = np.log10(df.iloc[:, 0:-1].div(df['length'], axis=0))

    return df, pc


def predict(word_counts, p_w_c, prior):
    """
    return argmax_c{p(c)mul_w{p(w|c)^N_w}}
    """
    res = (p_w_c.as_matrix() * word_counts).sum(axis=1) * prior.values
    best_class = prior.columns[res.argmax()]
    print(best_class)
    return best_class


def evaluate():
    voc = get_voc(1000, '20news/20news.voc')
    # p_w_c, prior = train(read_training_data('20news/20news.tr', voc))
    # test = read_training_data('20news/20news.te', voc)
    p_w_c, prior = train(read_training_data('spam/spam.tr', voc))
    test = read_training_data('spam/spam.te', voc)

    results = []
    for idx, row in test.iterrows():
        actual = row['class_name']
        pred = predict(row[1:].values, p_w_c, prior)
        results.append((actual, pred))
        # print(f'actual: {actual}, pred: {pred}')

    print('err:', len([1 for i in results if i[0] != i[1]]) / len(results))


if __name__ == '__main__':
    evaluate()
