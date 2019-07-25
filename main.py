from src.w2v import w2v
from src.config import config
import json

model = w2v(lang='zh', cache_path='data/nn_zh.jl')
model.load_stop('data/stop.txt')

data = []
cnt = 0

g = open('data/nsfc_kws_filt.jl', 'w', encoding='utf-8')

print('preprocessing documents...')

cnt = 0
with open('data/nsfc_kws.jl', 'r', encoding='utf-8') as f:
    for line in f:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt, end='\r')
            # break
        th = json.loads(line)
        th["words"] = [(x[0], round(x[1], 3)) for x in model.associate(th["words"], config.MIN_SIM, config.MAX_ASSO) if x[0] not in model.stop]
        g.write(json.dumps(th, ensure_ascii=False))
        g.write('\n')

g.close()
print(cnt, 'documents preprocessed over.')

import math
from collections import defaultdict
from collections import Counter
global_freq = Counter()
global_score = defaultdict(float)

print('calculating word scores...')

cnt = 0
with open('data/nsfc_kws_filt.jl', 'r', encoding='utf-8') as f:
    for line in f:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt, end='\r')
            # break
        th = json.loads(line)
        words = set([x[0] for x in th['words']])
        for x in th['words']:
            global_freq[x[0]] += 1
            global_score[x[0]] += x[1]
for k in global_score:
    global_score[k] /= global_freq[k]
print(cnt, 'documents calculated over.')

coo = Counter()
score_sig = defaultdict(float)
score_coo = defaultdict(lambda: defaultdict(float))
score = {}

print('finding hierarchy...')

cnt = 0
with open('data/nsfc_kws_filt.jl', 'r', encoding='utf-8') as f:
    for line in f:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt, end='\r')
            # break
        th = json.loads(line)
        new_words = []
        for x in th['words']:
            if global_score[x[0]] >= config.MIN_SCORE and global_freq[x[0]] >= config.MIN_FREQ:
                new_words.append(x)
                score_sig[x[0]] += x[1]
        for i in range(len(new_words)):
            for j in range(i+1, len(new_words)):
                pa = tuple(sorted([new_words[i][0], new_words[j][0]]))
                coo[pa] += 1
                score_coo[pa][new_words[i][0]] += new_words[i][1]
                score_coo[pa][new_words[j][0]] += new_words[j][1]

print(cnt, 'documents over.')

print('calculating confidence score...')

cnt = 0
for pa in score_coo:
    if coo[pa] < config.MIN_COOCCUR: continue
    cnt += 1
    if cnt % 100 == 0:
        print(cnt, end='\r')
    score[pa] = score_coo[pa][pa[0]] / score_sig[pa[0]] - score_coo[pa][pa[1]] / score_sig[pa[1]]

print(cnt, 'pairs calculated over.')

print('output to file...')

cnt = 0
g = open('data/result.txt', 'w', encoding='utf-8')
for pa in score:
    if abs(score[pa]) < config.MIN_CONF: continue
    cnt += 1
    if cnt % 100 == 0:
        print(cnt, end='\r')
    if score[pa] > 0:
        father = pa[1]
        child = pa[0]
        conf = round(score[pa], 3)
    else:
        father = pa[0]
        child = pa[1]
        conf = round(-score[pa], 3)
    g.write(pa[0] + '\t' + pa[1] + '\t{}'.format(conf) + '\n')

g.close()
print(cnt, 'hyponymy founded. output to data/result.txt')
