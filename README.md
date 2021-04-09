BERT NLP toolkit
=============

BERT NLP toolkit (https://pypi.org/project/bertnlp) is a Python package that performs various NLP tasks using Bidirectional Encoder Representations from Transformers (BERT) related models.

Installation
-------

To Install this package using pip:

```Shell
pip install bertnlp-0.0.x-py3-none-any.whl -f https://download.pytorch.org/whl/torch_stable.html
```

Implemented NLP Solutions
--------
* BERT tokenizer
* BERT word embedding and fuzzy matcher
* BERT sentence embedding
* Modified BERT sentiment score
* Text classifier based on KNN-bert and trainer
* Text classifier based on FastText and trainer
* Multi-labelled text intent detector based on FastText and trainer

Usage
--------
To use this bert package as a SDK, 
```python
from bertnlp.fuzzy_matcher import semanticMatcher
from bertnlp.pipeline import sentiment,embeddings,tokenizer
from bertnlp.text_classifier import knnbert as bert_clf

corpus = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?','cat','TV']


feature_list=['cat','dog','television','guitar','movie','pizza','pasta']

matcher=semanticMatcher()
sentimentScorer = sentiment(neu_range=0.2)
senti_pred = sentimentScorer.score(corpus)
for j, sent in enumerate(corpus):
    features=matcher.match_sent(sent, feature_list, threshold=0.3)
    feature_mentioned= ';'.join(['{:s}, score:{:.4f}'.format(f['label'],f['score'] ) for f in features])
    print("[Sentence] {:s}; [Sentiment] {:s}, score:{:.4f}; [Feature Mentioned] {:s}".format(
        sent,senti_pred[j]['label'],senti_pred[j]['score'],feature_mentioned)
    )

print('\n + Extra pipeline features added 12142020:')
senti_pred=sentimentScorer.predict(corpus)
senti_score=sentimentScorer.predict_proba(corpus)
bert_tok=tokenizer()
emb=embeddings()
# sentence bert embeddings: Input  - list of sentences, Output - 2D numpy array
sent_emb= emb.sbert_emb(corpus)
print('embedding shape:', sent_emb.shape)
for j, sent in enumerate(corpus):
    print(
        "[Sentence] {:s}; [Sentiment proba] :{:s};  [1-5th dimensions of sentence embedding] {:s}".format(
            sent, str(senti_score[j,:].tolist()), str(sent_emb[j,:5].tolist()) )
    )
    # bert word embeddings: Input  - list of words, Output - 2D numpy array
    tokens=bert_tok.token(sent)
    word_emb=emb.bert_emb(tokens)
    for i,tok in enumerate(tokens):
        print("[Token] {}; [1-5th dimension of Word Embeddings] {}".format(tok, word_emb[i,:5]) )

print('Embedding cosine similarity as text relevancy:')
print(emb.cos_sim(sent_emb[:3],sent_emb[-3:]))
```
To train a text classifier
```python
from bertnlp.text_classifier import knnbert,trainer
import numpy as np
from bertnlp.utils import get_example_data
from bertnlp.measure import plotConfMat


def drop_class(X,y,classname):
    sel_id=[i for i,yy in enumerate(y) if yy!=classname]
    return [X[i] for i in sel_id],[y[i] for i in sel_id]

data,senti_cat,subcat,senti_label,featureMent=get_example_data('train','ISO-8859-1')
test_data,test_senti_cat,test_subcat,test_senti_label,test_featureMent=get_example_data('test','ISO-8859-1')
sbert_model_name='roberta-base-nli-stsb-mean-tokens'

cat_data_tr,cat_tr=drop_class(data,senti_cat,"Critique")
cat_data_te, cat_te = drop_class(test_data, test_senti_cat, "Critique")
cat_model = knnbert(sbert_model_name=sbert_model_name)

# using evaluation mode to check training-validation performance stats
trainer(cat_data_tr+cat_data_te,cat_tr+cat_te,cat_model,eval_round=10)
# enable save_model_path to generate a deployable model on the overall dataset.
cat_model =trainer(cat_data_tr+cat_data_te,cat_tr+cat_te,cat_model,save_model_path='./model.pkl')
cat_pred=cat_model.predict(cat_data_te)
print('senti_cat sbert model test accuracy {}'.format((cat_pred==np.array(cat_te)).mean() ))
plotConfMat(cat_te,cat_pred,cat_model.classes_,'cat_sbertClf_confmat.png')
```
The bert package support multi-labelled text intent detection, which can adapted to multiple NLP tasks such as (1) intent detection, (2) multi-intent detection, and other text classification tasks.
Different from text classification, the multi-labelled text intent detection can (1) check 'None' class and (2) classify a single text into multiple labels as many as it detects. 
To train a multi-labelled text detection model,
```python
from bertnlp.text_classifier import fasttextClf
import numpy as np
from bertnlp.utils import get_example_data
from bertnlp.measure import plotPrecisionRecall
from bertnlp.fuzzy_matcher import feat_predict_func

# load the example data for training
data,senti_cat,subcat,senti_label,featureMent=get_example_data('train','ISO-8859-1')
test_data,test_senti_cat,test_subcat,test_senti_label,test_featureMent=get_example_data('test','ISO-8859-1')
# save all the Mentioned features as a list of feature names
featurelist=list(set(sum(test_featureMent,[])+ sum(featureMent,[])))


X_tr=[data[i] for i,f in enumerate(featureMent) if 'None' not in ' '.join(f)]
X_te=test_data

# To reduce data imbalance, drop None classes during training.
featMent_ftmodel=fasttextClf()
tr_featMent=[''.join(f) for f in featureMent if 'None' not in ' '.join(f)]
featMent_ftmodel.fit(X_tr,tr_featMent,lr=1.0,epoch=100,wordNgrams=2,loss='ova')
featMent_ftmodel.model.save_model('./featMent_ftClf.bin')

# mixing a edit-distance based text fuzzy matcher with the multi-intent detector to improve simple cases.
combinedModel={'ftmodel':featMent_ftmodel,'featlist':featurelist}

def predict_func(data,model,conf_th):
    return feat_predict_func(data,model,conf_th)[0]

feat_pred,_=feat_predict_func(X_te,combinedModel,0.2)
print(feat_pred)
test_featMent=[''.join(f) for f in test_featureMent]
print(test_featureMent)
# using test features to check performance stats: Precision-Recall are used
best_prec_rec,best_thresh=plotPrecisionRecall(X_te,test_featureMent,combinedModel, predict_func, \
                                              conf_thresh_range=np.arange(0,1,0.1),fig_path= './aspect_ftClf_prec_rec.png')

print('The best Precision- Recall is evaluated at confident threshold {}:'.format(best_thresh))
print('Precision: ',best_prec_rec['precision'],'Recall: ',best_prec_rec['recall'])
for c,rate in best_prec_rec['prec_by_class'].items():
    print('class',c,': ',rate)
``` 
