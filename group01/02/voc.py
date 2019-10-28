from dictionary import BidirectionalDict
import re

d = BidirectionalDict()
with open('Europal-v9', 'r') as f:
    cnt = 0
    for line in f.readlines():
        for word in line.strip().split(' '):
            if not re.match(r'[\w\d]+', word):
                continue
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
        cnt += 1

        # if cnt >= 10000:
        #     break

kv = sorted(
    [(item.key, item.value) for node in d.kv.nodes for item in node.items if item],
    key=lambda x: x[1], reverse=True
)
print(kv[:100])
