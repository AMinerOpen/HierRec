from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict
import json

class w2v:
    def __init__(self, lang='en', model_path=None, model=None, cache_path=None):
        assert lang in ['en', 'zh']
        print('loading model...')
        if model_path is None:
            model_path = 'tmp/keywords_aminer_{}'.format(lang)
        if model is None:
            self.model = KeyedVectors.load(model_path).wv
        else:
            self.model = model
        self.model.most_similar("数据挖掘")
        self.lang = lang
        self.nn_cache = {}
        if cache_path is not None:
            with open(cache_path, 'r') as f:
                for line in f:
                    th = json.loads(line)
                    self.nn_cache[th['word']] = th['nn']
    def load_stop(self, stop_path):
        self.stop = set()
        with open(stop_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stop.add(line.strip())
    def filter(self, words, MIN_SCORE):
        return [x for x in words if x['weight'] >= MIN_SCORE]
    def shrink(self, words):
        w2w = defaultdict(float)
        for x in words:
            w2w[x[0]] += x[-1]
        return [x for x in sorted(w2w.items(), key=lambda x:x[-1], reverse=True)]
    def restrict(self, words):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []
        new_vectors_norm = []
        for i in range(len(self.model.vocab)):
            word = self.model.index2entity[i]
            vec = self.model.vectors[i]
            vocab = self.model.vocab[word]
            vec_norm = self.model.vectors_norm[i]
            if word in words:
                vocab.index = len(new_index2entity)
                new_index2entity.append(word)
                new_vocab[word] = vocab
                new_vectors.append(vec)
                new_vectors_norm.append(vec_norm)
        self.model.vocab = new_vocab
        self.model.vectors = np.array(new_vectors)
        self.model.index2entity = np.array(new_index2entity)
        self.model.index2word = np.array(new_index2entity)
        self.model.vectors_norm = np.array(new_vectors_norm)
    def get_nn(self, w, topn, MIN_SIM):
        if w not in self.nn_cache or len(self.nn_cache[w]) < topn:
            self.nn_cache[w] = self.model.most_similar(w, topn=topn)
        return [i for i in self.nn_cache[w] if i[1] >= MIN_SIM]
    def associate(self, words, MIN_SIM, MAX_ASSO):
        ext = []
        for x in words:
            ext.extend([x] + [(i[0], x[-1]*i[1]) for i in self.get_nn(x[0], topn=MAX_ASSO, MIN_SIM=MIN_SIM)])
        return self.shrink(ext)

