from collections import defaultdict, Counter, deque
import random


class NGramModel:
    def __init__(self, n):
        self.n = n

        # models[n] are the actual model, others are for backing off
        self.models = {i+1: defaultdict(lambda: defaultdict(int)) for i in range(n)}
        self.models[1] = defaultdict(int)

        self.vocabulary = set()

    def update(self, sequence):
        assert len(sequence) <= self.n
        self.vocabulary.update(sequence)
        h, w = sequence[:-1], sequence[-1]

        # update n-gram models where n >= 2
        for n in range(len(sequence), 1, -1):
            self.models[n][tuple(h)][w] += 1
            h.pop(0)

        # update 1-gram model
        self.models[1][w] += 1
    
    def count_of_counts(self, turing_good_smoothing=False):
        result = Counter()

        model = self.models[self.n]
        for h, words in model.items():
            result.update(words.values())

        return result if not turing_good_smoothing else self._turing_good_smooting(result)

    def count_of_counts_of_prefix(self, h, turing_good_smoothing=False):
        """
        return a dict {r: N_r(h, *)}
        """
        result = Counter()

        counts = self._get_counts_of_prefix(h)
        if counts:
            result.update(counts.values())

        return result if not turing_good_smoothing else self._turing_good_smooting(result)

    def _turing_good_smooting(self, counter):
        """
        Smooth all counts in counter using turing good method
        """
        result = Counter()
        for r, n_r in counter.items():
            if n_r != 0:
                result[r] = (r + 1) * counter[r+1] / n_r
            else:
                result[r] = n_r

        return result

    def vocabulary_size(self):
        return len(self.vocabulary)
    
    def N_h_w(self, h, w):
        """
        return N(h, w) of the model. Back off to lower-gram if h not seen in current gram.
        """
        counts = self._get_counts_of_prefix(h)
        if counts:
            return counts[w]
        return 0

    
    def N_h(self, h):
        """
        return N(h, *) of the model. Back off to lower-gram if h not seen in current gram.
        """
        counts = self._get_counts_of_prefix(h)
        if counts:
            return sum(counts.values())
        return 0
    
    def _get_counts_of_prefix(self, h):
        """
        Starting from (self.n)-gram model, automatically back off to next lower-gram model
        if h not found in current gram.

        return a dict containing counts {w: N(h, w)}
        """
        assert len(h) + 1 == self.n
        h = h.copy()

        for n in range(self.n, 1, -1):
            prefix = tuple(h)
            if prefix in self.models[n]:
                result = self.models[n][prefix]

                # set N(h,w)=0 if hw not seen in corpus
                if not len(result.keys()) == len(self.vocabulary):
                    not_seen = self.vocabulary - set(result.keys())
                    for w in not_seen:
                        result[w] = 0

                return result

            # cut h and back off to (n-1)-gram model
            h.pop(0)


def train_ngram_model(n=2):
    """
    Train a n-gram model
    """
    model = NGramModel(n)
    with open('Europal-v9', 'r') as f:
        cnt = 0
        sequence = deque(maxlen=n)

        print(f'Training {n}-gram model...')

        for line in f.readlines():
            for word in [word for word in line.strip().split(' ') if len(word) > 0]:
                sequence.append(word)
                model.update(list(sequence))

            cnt += 1
            if cnt >= 10000:
                break

        print(f'Training finished')
    return model


# ================== abs discounting =============
def get_params(model, h):
    """
    Prepare params for absolute discounting
    """
    coc_h = model.count_of_counts_of_prefix(h)
    coc = model.count_of_counts()

    # b_h = coc_h[1] / (coc_h[1] + 2*coc_h[2])
    b = coc[1] / (coc[1] + 2*coc[2])
    # b = 0.96
    N_h = model.N_h(h)

    sum_beta = 0
    for v in model.vocabulary:
        sum_beta += model.N_h_w(h, v) / N_h

    return b, N_h, sum_beta, coc_h


def absolute_discounting(model, h, w, N_h, coc_h, b_h, sum_beta):
    N_hw = model.N_h_w(h, w)
    W = model.vocabulary_size()

    if N_hw > 0:
        return (N_hw - b_h) / N_h

    beta_hw = N_hw / N_h
    return b_h * ((W - coc_h[0]) / N_h) * (beta_hw / sum_beta)


def interpolated_absolute_discounting(model, h, w, N_h, coc_h, b_h, sum_beta):
    N_hw = model.N_h_w(h, w)
    W = model.vocabulary_size()

    if N_hw > 1:
        return (N_hw - b_h) / N_h

    beta_hw = N_hw / N_h
    res = max(N_hw - b_h, 0) / N_h + b_h * ((W - coc_h[0]) / N_h) * (N_hw / N_h)
    return res

# =========================================================


def get_sentence(model, start=['I'], length=20, discounting_method=absolute_discounting):
    assert len(start) + 1 == model.n,\
        f"Your are using {model.n}-gram model, must provide {model.n-1} words as start."

    result = start
    h = deque(start, maxlen=model.n-1)

    for _ in range(length):
        b_h, N_h, sum_beta, coc_h = get_params(model, list(h))
        probs = [
            (w, discounting_method(model, list(h), w, N_h, coc_h, b_h, sum_beta)) 
            for w in model.vocabulary
        ]
        import ipdb; ipdb.set_trace()
        probs = [(w, prob) for w, prob in probs if prob > 0]
        probs.sort(key=lambda x: x[1])

        # randomly choose from pick a word from top3 most possible ones
        next_word = random.choice(probs[-3:])[0]
        print(f'{probs[-3:]} -> {next_word}')

        result.append(next_word)
        h.append(next_word)

    print(f"Generated sencence:\n{' '.join(result)}")
    return result


def ex5():
    n = 2
    # a)
    model = train_ngram_model(n=n)
    # b)
    coc = model.count_of_counts()
    print(f'count of counts without smoothing:\n{coc}')
    # c)
    coc_turing_good = model.count_of_counts(turing_good_smoothing=True)
    print(f'count of counts with turing good smoothing:\n{coc_turing_good}')

    # e)
    n = 3
    model = train_ngram_model(n=n)
    print('Sentence generated with absolute discounting:')
    get_sentence(model, start=['I', 'would'], length=30, discounting_method=absolute_discounting)

    print('\nSentence generated with interpolated absolute discounting:')
    get_sentence(model, start=['I', 'would'], length=30, discounting_method=interpolated_absolute_discounting)


if __name__ == "__main__":
    ex5()
