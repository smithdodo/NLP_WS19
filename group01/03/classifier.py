from collections import namedtuple, Counter
import numpy as np
import matplotlib.pyplot as plt



WordCount = namedtuple('WordOccurrences', field_names=['word', 'count'])
X = namedtuple('X', field_names=['c', 'N', 'N_w'])


voc_path = '20news/20news.voc'
training_set_path = '20news/20news.tr'
test_set_path = '20news/20news.te'


voc = []
training_set = []
test_set = []


def get_voc(size):
    with open(voc_path, 'r') as f:
        all_vocs = [w.strip() for w in f.read().split('\n')]
        limit = size if size < len(all_vocs) else len(all_vocs)
        voc = {v.strip():0 for v in all_vocs[:limit]}
    return voc



def read_data_set(path, vocabulary):
    res = []
    with open(path, 'r') as f:
        for line in f.readlines():
            elements = line.strip().split(' ')
            class_ = elements[1]
            occurrences = [WordCount(word=wn[0], count=int(wn[1])) for wn in zip(elements[2::2], elements[3::2])]
            N = sum([wn[1] for wn in occurrences])
            rel_freq = [(wc.word, wc.count/N, wc.count) for wc in occurrences if wc.word in vocabulary]
            res.append(X(c=class_, N=N, N_w=rel_freq))

    return res


def get_class_prior(dataset):
    """
    return p(c) for all classes
    """
    class_cnt = Counter([x[0] for x in dataset])
    total_cnt = sum(class_cnt.values())
    pc = {k:v/total_cnt for k,v in class_cnt.items()}
    return pc


def get_p_w_c(dataset, vocabulary):
    """
    return p(w|c) for all classes
    """
    p_w_c = {}
    for x in dataset:
        class_, doc_length, frequencies = x
        if class_ not in p_w_c:
            p_w_c[class_] = {'voc': vocabulary.copy(), 'doc_length': 0}
        p_w_c[class_]['doc_length'] += doc_length

        for freq in frequencies:
            word, count, _ = freq
            if word in p_w_c[class_]['voc']:
                p_w_c[class_]['voc'][word] += count

    for class_, item in p_w_c.items():
        # convert word count to relative frequency
        p_w_c[class_]['voc'] = {word: count/item['doc_length'] for word, count in item['voc'].items()}
    
    return p_w_c


def train(training_set_path, vocabulary):
    """
    return p(c) p(w|c) for all classes
    """
    training_set = read_data_set(training_set_path, vocabulary)
    # p(c)
    pc = get_class_prior(training_set)
    # p(w|c)
    p_w_c = get_p_w_c(training_set, vocabulary)
    return pc, p_w_c


b = 0
def get_class_posterior(class_, word_frequencies, pc, p_w_c):
    """
    params:
        class: c
        word_frequences: N_1^w, input word sequence with relative frequency
    return p(c|N_1^w)
    """
    global b
    a = pc[class_] * np.prod([p_w_c[class_]['voc'][word] ** count for word, _, count in word_frequencies])
    if b == 0:
        b = np.sum([
            pc[c] * np.prod([p_w_c[c]['voc'][word] ** count for word, _, count in word_frequencies])
            for c in pc.keys()
        ])
    return a/b


def predict(word_frequencies, pc, p_w_c):
    """
    return argmax_c{p(c|N_1^w)}
    """
    probabilities = [
        [c, get_class_posterior(c, word_frequencies, pc, p_w_c)] for c in pc.keys()
    ]
    probabilities.sort(key=lambda x: x[1], reverse=True)
    print(f'highest prob: {probabilities[0][0]}, {probabilities[0][1]}')
    return probabilities[0][0]


def plot():
    xs = []
    ys = []
    max_voc = 10000
    for voc_size in range(1, max_voc, 100):
        voc = get_voc(voc_size)
        pc, p_w_c = train(training_set_path, voc)
        test_set = read_data_set(test_set_path, voc)
        predictions = [(ts.c, predict(ts.N_w, pc, p_w_c)) for ts in test_set]
        error_rate = len([None for x in predictions if x[0] != x[1]]) / len(test_set)
        print('error rate:', error_rate)
        xs.append(voc_size)
        ys.append(error_rate)
    
    plt.title('title')
    plt.ylabel('error rate')
    plt.xlabel('voc size')
    plt.plot(xs, ys)
    plt.savefig(f'plt_{max_voc}.png')
    plt.show()

plot()
