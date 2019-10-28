import numpy as np
import re
import matplotlib.pyplot as plt


with open('Europal-v9', 'r') as f:
    content = [l.strip() for l in f.readlines()]


sentence_lentghs = []
word_lengths = []
for line in content:
    sentence_lentghs.append(len(line.split(' ')))
    # we split each line with space, filter out those words consists with only non-character 
    # chars. In the end we remove non-character chars from each word and count their lengths.
    word_lengths += [
        len(re.sub(r'\W+', '', w)) 
        for w in line.split(' ') if not re.match(r'\W+', w) and w != '_'
    ]


np_s_len = np.array(sentence_lentghs)
np_w_len = np.array(word_lengths)


print(f'sentences mean: {np_s_len.mean()}, variance: {np_s_len.var()}')
# sentences mean: 28.190896666666667, variance: 291.8349484626555
print(f'words mean: {np_w_len.mean()}, variance: {np_w_len.var()}')
# words mean: 4.866916678844433, variance: 8.139043363701868


fig, (ax1, ax2) = plt.subplots(2)
ax1.set_title('sentence lengths distribution')
ax1.hist(np_s_len, bins=30)
ax2.set_title('word lengths distribution')
ax2.hist(np_w_len, bins=len(set(np_w_len)))
plt.show()
