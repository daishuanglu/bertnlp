# Standard libraries
from collections import Counter
import os

# Third party libraries
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
from scipy.sparse.csgraph import csgraph_from_dense
import re
from fasttext import train_supervised,load_model
from tabulate import tabulate
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report


stats_names=['precision','recall','f1-score','support']

def keep_alpha(s):
    return re.sub(r"[^A-Za-z']+", '-', s)

def norm01( x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def merge_multi_line_text(x):
    print(x)
    lines = x.split('\n')
    result = ''
    for line in lines:
        if line.isspace() or len(line) < 1:
            continue

        # remove trailing space:
        line = line.rstrip()
        if line.endswith('.') or line.endswith('!') or line.endswith('?'):
            result += line + ' '
        else:
            result += line + '. '
    return result


class knnbert():

    def __init__(self, sbert_model_name='roberta-base-nli-stsb-mean-tokens',pretrained=''):
        self.emb_model_name=sbert_model_name
        self.embeddings = None
        self.classes_=None
        if pretrained:
            self.load_model(pretrained)
        else:
            self.sbert_model = SentenceTransformer(sbert_model_name)
            self.knn = None
        self.ext_features = None


    def _inputs(self,X_emb,ext_features=None):
        if ext_features is None:
            return X_emb
        else:
            return torch.hstack((X_emb,torch.from_numpy(ext_features)))


    def predict(self,X,ext_features=None):
        probs=self.predict_proba(X,ext_features)
        ind=np.argmax(probs,axis=1)
        return self.classes_[ind]


    def predict_proba(self,X,ext_features=None):
        X_embeddings = self.sbert_model.encode(X, convert_to_tensor=True)
        X_ext = self._inputs(X_embeddings, ext_features)
        if self.embeddings is not None:
            model_X_ext = self._inputs(self.embeddings, self.ext_features)
            cosine_scores = util.pytorch_cos_sim(X_ext, model_X_ext)
            G2_sparse = 2 - cosine_scores.numpy()
            # G2_sparse = csgraph_from_dense( 2-cosine_scores.numpy(), null_value=np.inf)
            probs = self.knn.predict_proba(G2_sparse)
        else:
            X_ext = self._inputs(X_embeddings, ext_features)
            probs = self.knn.predict_proba(X_ext)
        return probs


    def fit(self, X, y, ext_features=None, sbert_sim=True, knn=10):
        """
        # if the number of training samples is greater than 100, we use StratifiedKFold to split it into 100 folds
        # else, split it into number of samples of folds
        counter = Counter(y)
        least_common = counter.most_common()[-1]
        _, value = least_common

        if (value > 100):
            my_cv = 100
        else:
            my_cv = max(value, 2)
        print(f'The number of folds on cross validation is: {my_cv}')

        # create a dictionary of all values we want to test for n_neighbors

        param_grid = dict(n_neighbors=range(1, knn,2))
        print(f'Start to do parameter sweeping on the grid {param_grid}')
        """

        self.embeddings = self.sbert_model.encode(X, convert_to_tensor=True)
        self.ext_features=ext_features
        X_ext=self._inputs(self.embeddings,ext_features)

        if sbert_sim:
            cosine_scores = util.pytorch_cos_sim(X_ext, X_ext)
            inputs = csgraph_from_dense( 2-cosine_scores.numpy(), null_value=np.inf)
            self.knn = KNeighborsClassifier(metric='precomputed')
        else:
            inputs =X_ext
            self.knn = KNeighborsClassifier()
            self.embeddings=None

        #knn_gscv = GridSearchCV(knn, param_grid, cv=my_cv)
        #knn_gscv.fit(inputs,y)
        self.knn.fit(inputs, y)
        self.classes_=self.knn.classes_
        #score_of_best_model = knn_gscv.best_estimator_.score(inputs, y)
        #print("best score on training data for knn classifier:" + str(score_of_best_model))

        #print("best score on cross validation for knn classifier:" + str(knn_gscv.best_score_))
        #self.best_classifier = knn_gscv.best_estimator_
        return self


    def save_model(self,model_path):
        file= os.path.join( model_path)
        f = open(file, 'wb')
        model = {'clf': self.knn, 'sbert_model_name': self.emb_model_name,'classes':self.classes_,\
                 'emb':self.embeddings,'ext_features':self.ext_features}
        torch.save(model, f)
        f.close()
        return

    def load_model(self,model_path):
        model = torch.load(open(model_path, 'rb'))
        self.sbert_model = SentenceTransformer(model['sbert_model_name'])
        self.knn = model['clf']
        self.embeddings = model['emb']
        self.ext_features=model['ext_features']
        self.classes_=model['classes']
        return self


class fasttextClf():

    def __init__(self,pretrained='',code='utf-8'):
        self.encoding = code
        self.model=None
        if pretrained:
            self.model=load_model(pretrained)
        self.train_file='fasttext.train'
        self.valid_file='fasttext.valid'

    def _prep_data(self,data,filePath):
        f=open(filePath,'w',encoding=self.encoding)
        for X,y in data:
            if isinstance(y,list):
                labelstr=' '.join(['__label__'+l for l in y])
            else:
                labelstr = '__label__'+ str(y)

            f.write(labelstr+' '+X+'\n')
        f.close()
        return

    def predict_sent(self,text,k=-1,conf_thresh=0.5):
        pred,score=self.model.predict(text,k=k,threshold=conf_thresh)
        return [p.replace('__label__','') for p in pred]


    def predict(self,corpus,k=-1,conf_thresh=0.3):
        ftpred = []
        for sent in corpus:
            pred = self.predict_sent(sent, k=k,conf_thresh=conf_thresh)
            if pred:
                ftpred.append(pred[0])
            else:
                ftpred.append('Other')
        return np.array(ftpred)


    def predict_proba(self,text,k=-1):
        pred,score=self.model.predict(text,k=k,threshold=0.0)
        return {pred[i].replace('__label__',''):score[i] for i in range(len(pred))}

    def fit(self,X,y,lr=0.5, epoch=25, wordNgrams=2, bucket=200000,loss='softmax'):
        self._prep_data(zip(X,y),self.train_file)
        self.model = train_supervised(input=self.train_file, lr=lr, epoch=epoch, \
                                      wordNgrams=wordNgrams, bucket=bucket, loss=loss)
        return self.model


    def valid(self,X,y,k=-1):
        self._prep_data(zip(X,y),self.valid_file)
        return self.model.test(self.valid_file,k=k)

    def fit_autoTune(self,Xtr,ytr,Xte,yte,tuneClass='',duration=600):
        self._prep_data(zip(Xtr, ytr), self.train_file)
        self._prep_data(zip(Xte, yte), self.valid_file)
        if tuneClass:
            self.model = train_supervised(input=self.train_file, autotuneValidationFile=self.valid_file,\
                                          autotuneMetric="f1:__label__"+tuneClass,autotuneDuration=duration)
        else:
            self.model = train_supervised(input=self.train_file, autotuneValidationFile=self.valid_file,\
                                      tuneDuration=duration)
        return self.model


def trainer(data,target,model,save_model_path='',eval_round=10,test_size=0.1,ftparams={}):
    X=np.asarray(data)
    label_names=sorted(set(target))
    if save_model_path:
        if ftparams:
            model.fit(X, target,lr = ftparams['lr'], \
                      epoch = ftparams['epoch'], wordNgrams =ftparams['wordNgrams'])
        else:
            model.fit(X, target)
        try:
            model.save_model(save_model_path)
        except:
            torch.save(model,save_model_path)

        return model
    else:
        #mse=0.0
        accuracy=0.0
        avg_stats = {l: {s: 0.0 for s in stats_names} for l in label_names}
        for r in range( eval_round):
            X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = test_size)
            model_r=model
            model_r.fit(X_train, y_train)
            y_pred = model_r.predict(X_test)
            #mse+=mean_squared_error(y_test, y_pred) ** 0.5
            accuracy+= accuracy_score(y_test,y_pred)
            stats_r = classification_report(y_test, y_pred , target_names=label_names, output_dict=True)
            for l in label_names:
                for s in stats_names:
                    avg_stats[l][s] += stats_r[l][s]

    for l in label_names:
        for s in stats_names:
            avg_stats[l][s] /= eval_round
    print('{:d} Training Round, AVG hard accuracy: {:.4f}'.format(eval_round,accuracy / eval_round))
    print('average STATS: \n')
    showStats(avg_stats)

    return


def showStats(stats):
    table=[['']+stats_names]+[['class '+str(l)]+[ '{:d}'.format(int(stats[l][s])) if s=="support" else '{:.2f}'.format(stats[l][s]) \
                                                  for s in stats_names] for l in stats.keys()]
    print(tabulate(table))
    return


def ft_predict_func(data,model,conf_th):
    ftpred = []
    for sent in data:
        pred = model.predict(sent, conf_thresh=conf_th)[0]
        if pred:
            ftpred.append(pred)
        else:
            ftpred.append(['None'])
    return ftpred


