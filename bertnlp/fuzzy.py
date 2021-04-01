import numpy as np
import re
import time
from nltk import ngrams
from fastDamerauLevenshtein import damerauLevenshtein
#from scipy.stats import wasserstein_distance


def keep_alpha(s):
    return re.sub(r"[^A-Za-z.']+", ' ', s)


def split_to_sentences(string):
    return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",string)


def extract_unique_ngrams(s,n):
    def checkLen(ws):
        for w in ws:
            if len(w)<=1:
                return False
        return True

    s=s.replace(" ' ",' ').split()
    if len(s)<n:
        return set(s)
    n=max(n,1)
    grams=[]
    for i in range(1,n+1):
        if s:
            grams+= [" ".join(w) for w in ngrams(s,i) if checkLen(w)]
    return set(grams)


class buckets():
    def __init__(self,custom_bucket={}):
        self.wlens = set([len(kw.split()) for category in custom_bucket for kw in custom_bucket[category]])
        self.words_bucket=custom_bucket

    def custom_buckets(self,sentence ):
        wlist =sentence.lower().split()
        wlist_ngrams={l:list(ngrams(wlist,l)) for l in self.wlens}

        similarity=[]
        for category in self.words_bucket.keys():
            sim=max([damerauLevenshtein(' '.join(w), kw.lower(), similarity=True) for kw in self.words_bucket[category] for w in wlist_ngrams[len(kw.split())]])
            similarity.append(sim)

        return similarity


class fuzzy_CVT():

    def __init__(self,sim_th=0.5,num_grams=2):
        self.sim_th = sim_th
        self.vocab=None
        self.num_grams=num_grams

    def fit(self,corpus,verbose=False):
        start_time=time.time()
        vocabulary = set()
        for i, paragraph in enumerate(corpus):
            words = set().union(*[extract_unique_ngrams(keep_alpha(sent.lower()), self.num_grams) \
                                  for sent in split_to_sentences(paragraph)])
            vocabulary = vocabulary.union(words)
            if verbose:
                print("processed {} truth vocab, {} secs".format(i + 1, int(time.time() - start_time)))
        self.vocab = sorted(self._fuzzy_vocab(list(vocabulary)))
        print('vocabulary length:', len(self.vocab))

        return

    def vectorize(self,corpus,continuous=True,verbose=False):
        start_time=time.time()
        vect = []
        for i, paragraph in enumerate(corpus):
            words = set().union(*[extract_unique_ngrams(keep_alpha(sent.lower()), self.num_grams) \
                                  for sent in split_to_sentences(paragraph)])
            vect.append(self._fuzzy_cvt(words, continuous))
            if verbose:
                print("processed {} text vectors, {} secs".format(i + 1, int(time.time() - start_time)))
        vect = np.stack(vect)
        return vect


    def _fuzzy_vocab(self,words):
        # input:
        # words - list of words
        reduced_words = []
        while words:
            w = words.pop(0)
            if len(reduced_words) == 0:
                reduced_words.append(w)
            if all([damerauLevenshtein(w, w1, similarity=True) < min(self.sim_th+len(w1)*0.01,0.99) for w1 in reduced_words]):
                reduced_words.append(w)
        return reduced_words


    def _fuzzy_cvt(self,words,continuous=True):
        vect=np.zeros(len(self.vocab))
        for word in words:
            if continuous:
                vect+=np.array([damerauLevenshtein(w,word,similarity=True) for w in self.vocab])
            else:
                vect+=np.array([damerauLevenshtein(w,word,similarity=True)> min(self.sim_th+len(w)*0.01,0.99) for w in self.vocab])
        return vect


    def disp_vocab(self):
        for v in self.vocab:
            print(v)
        return

