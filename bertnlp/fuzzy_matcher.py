# Third-party libraries
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer,util
from nltk import pos_tag,ngrams
import re
from fastDamerauLevenshtein import damerauLevenshtein

from numpy import dot
from numpy.linalg import norm


def vec_cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def keep_alpha(s):
    return re.sub(r"[^A-Za-z0-9']+", '', s)


def pairwise_euclid(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def pairwise_cosine(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = x / x.norm(dim=1)[:, None]
    y_norm = y / y.norm(dim=1)[:, None]
    dist = torch.mm(x_norm, y_norm.transpose(0, 1))
    return dist


class semanticMatcher():

    def __init__(self,sent_model_name='roberta-base-nli-stsb-mean-tokens',token_model_name="bert-base-uncased"):
        self.sbert_model = SentenceTransformer(sent_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(token_model_name)
        self.bert_model = BertModel.from_pretrained(token_model_name, return_dict=True)


    def sent_similarity(self,sentences1,sentences2):
        # Input:
        #   sentences1: list of string - a list of sentences
        #   sentences2: list of string - a list of another sentences
        # output:
        #   cosine_scores: list of float numbers - cosine similarity.
        
        embeddings1 = self.sbert_model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.sbert_model.encode(sentences2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_scores.numpy()


    def sent_paraphrase(self,sentences):
        # Input:
        #   sentences: list of string - a list of sentences
        # output:
        #   paraphrases: list of tuple - a list of [score, i,j] tuple
        paraphrases = util.paraphrase_mining(self.sbert_model, sentences)

        return paraphrases


    def match_sent(self,sentence,target_list,threshold=0.3):

        sim=self.sent_similarity([sentence],target_list)[0]
        matched=[ {'label':target_list[i],'score':s} for i,s in enumerate(sim) if s>threshold]
        return matched


class tokenMatcher():

    def __init__(self,model_name='bert-base-uncased',tags=['NN']):
        self.check_tags=tags
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name, return_dict=True)

    def _get_bert_embeddings(self,string):
        inputs = self.tokenizer(string, return_tensors="pt")
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[0,1:-1,:].detach()


    def find_code(self,sentence,hard_codes):
        targets=[w.lower() for w in hard_codes]
        matched=[]
        for word,tag in pos_tag(self.tokenizer.tokenize(sentence)):
            if any([tag in t for t in self.check_tags]):
                w=keep_alpha(word).lower()
                if w in targets:
                    matched.append( {'label': hard_codes[targets.index(w)],'score':1.0 })
        return matched


    def token_similarity(self,sentence,target_list):
        tokens_emb=self._get_bert_embeddings(sentence)
        targets_emb = torch.stack([self._get_bert_embeddings(target)[0] for target in target_list])
        return pairwise_cosine(tokens_emb, targets_emb).numpy()


    def match_token(self,sentence,target_list,threshold):
        sim= self.token_similarity(sentence,target_list)

        matched = []
        for irow in range(sim.shape[0]):
            ind, = np.where(sim[irow, :] > threshold)
            matched+=[{'label': target_list[icol], 'score': sim[irow, icol].item()} for icol in ind]
        return matched


def merge_matched_terms(terms1,terms2):
    terms1=list( map(list,zip(*[(t['label'],t['score']) for t in terms1])))
    terms2 =list( map(list, zip(*[(t['label'], t['score']) for t in terms2])))
    merged_terms =[]
    for i,t in enumerate(terms1[0]):
        term={'label':t,'score':terms1[1][i]}
        if t in terms2[0]:
            term['score'] = (term['score']+ terms2[1][terms2[0].index(t)])/2
        merged_terms.append(term)
    return merged_terms


def find_code_in_sent(sentence,hard_codes):
    sent=sentence.lower()
    matched=[]
    words = sent.split()
    bigrams=[]
    if len(words)>0:
        bigrams = [' '.join(w) for w in ngrams(words, 2)]
    for t in hard_codes:
        word_sim=max(flu_score(words,[t.lower()],overlap=False)[0])
        if word_sim>0.5:
            matched.append({'label': t,'score':word_sim })
        if len(bigrams)==0:
            continue
        bigram_sim = max(flu_score( bigrams, [t.lower()], overlap=False)[0])
        if bigram_sim>0.5:
            matched.append({'label': t,'score':bigram_sim  })
        #edit_sim =  max([damerauLevenshtein(w, t, similarity=True) for w in words])
        #if edit_sim>0.5:
        #    matched.append({'label': t, 'score': edit_sim})
    return matched


def find_code_fast(sentence,hard_codes,th=0.7):
    wlist=sentence.lower().split()
    kwlens=[min(len(t.split()),len(wlist)) for t in hard_codes]
    if len(wlist)==0:
      return []
    wlist_ngrams={l:list(ngrams(wlist,l)) for l in set(kwlens)}
    matched = []
    for i,kw in enumerate(hard_codes):
        sim=max([damerauLevenshtein(''.join(w).lower(), kw.replace(' ','').lower(), similarity=True)  for w in wlist_ngrams[kwlens[i]]])
        if sim>th:
            matched.append({'label': kw, 'score': sim})
    return matched


def emb_score(strlist, targetlist,emb):
    embstr=[emb[s] for s in strlist]
    embtar=[emb[s] for s in targetlist]
    return np.array([[vec_cos_sim(es,et) for es in embstr] for et in embtar])


def get_ngram(text: str, size: int) -> set:
    """
    Generate NGrams from string
    string -> {'st','tr','ri','in','ng'}
    :param text: string/text
    :param size: ngram size
    :return: ngram set
    """
    ngs = set({})
    for w in set(text.lower().split()):
        for ngram in ngrams(w, size):
            ngs.add(''.join(ngram))
    return ngs


def jaccard(set1: set, set2: set) -> float:
    """
    calculate jaccard similarity between two sets
    jaccard = (set1 intersect set2)/(set1 union set2)
    set1 = {'st','tr','ri','in','ng'} and set2 = {'st','tr','ri','in'}
    then jaccard will be 4/5 = 0.8
    :param set1:
    :param set2:
    :return:
    """
    numerator = len(list(set1.intersection(set2)))
    denom = len(list(set1.union(set2)))
    if denom == 0:
        return 0
    score = numerator / denom
    return round(score, 2)


def jaccard_ext(set1: set, set2: set) -> float:
    """
    calculate jaccard similarity between two sets
    jaccard = (set1 intersect set2)/(set1 union set2)
    set1 = {'st','tr','ri','in','ng'} and set2 = {'st','tr','ri','in'}
    then jaccard will be 4/5 = 0.8
    :param set1:
    :param set2:
    :return:
    """
    numerator = len(list(set1.intersection(set2)))
    denom = len(list(set1.union(set2)))
    min_len = min(len(set1), len(set2))
    if denom == 0:
        return 0
    score = 0.5 * ((numerator / denom) + (numerator / min_len))
    return round(score, 2)


def calculate_jaccard(str1: str, str2: str, ngram_size: int = 2,overlap:bool = False) -> float:
    """
    calculate jaccard similarity between two strings
    :param str1: string1
    :param str2: string2
    :param ngram_size: ngram size
    :return:
    """
    str1 = '_' + str1.replace(' ', '_') + '_'
    str2 = '_' + str2.replace(' ', '_') + '_'

    set1 =  get_ngram(str1, size=ngram_size)
    set2 =  get_ngram(str2, size=ngram_size)

    if overlap:
        score =  jaccard_ext(set1=set1, set2=set2)
    else:
        score =  jaccard(set1=set1, set2=set2)
    return score


def flu_score(strlist, targetlist,overlap=True,inter=True) :
    """
    Calculate the jaccard score
    :param strlist: list of string - list of dottedDict
    :param targetlist:  input - dottedDict
    :return: list of list: 2d similarity matrix (i,j) indicates the i-th and j-th similarity
    """
    if inter:
        return np.array([[calculate_jaccard(str1=s, str2=t, overlap=overlap) for s in strlist] for t in targetlist])
    else:
        assert(len(strlist)==len(targetlist))
        return np.array([calculate_jaccard(str1=strlist[i], str2=targetlist[i], overlap=overlap) for i in range(len(strlist))])


def feat_predict_func(data,model,conf_th):
    feat_pred=[]
    feat_score=[]
    for i,sent in enumerate(data):
        term=find_code_fast(sent,model['featlist'])
        term_ft ={k:v for k,v in model['ftmodel'].predict_proba(sent).items() if v>conf_th}
        for t in term:
            if t['label'] in term_ft.keys():
                term_ft[t['label']]=(term_ft[t['label']]+ t['score'])/2
            else:
                term_ft.update({t['label']:t['score']})
        if term_ft:
            feat_pred.append(list(term_ft.keys()))
            feat_score.append(list(term_ft.values()))
        else:
            feat_pred.append(['None'])
            feat_score.append([1.0])
    return feat_pred,feat_score
