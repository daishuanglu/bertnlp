# Third-party libraries
from transformers import pipeline, BertTokenizer, BertModel
import numpy as np
from sentence_transformers import SentenceTransformer,util
import torch

class sentiment():

    def __init__(self,neu_range=0.15):
        self.neu_range=neu_range
        self._classnames=['NEGATIVE','POSITIVE']
        self.crafted_classnames = ['NEGATIVE','NEUTRAL','POSITIVE']
        return

    def score(self,sentences):
        classifier = pipeline('sentiment-analysis')
        pred=[]
        for result in classifier(sentences):
            if (result['score']>0.5-self.neu_range) and (result['score']<0.5+self.neu_range):
                result['label']='NEUTRAL'
            if result['label']=='NEGATIVE':
                result['score']=1-result['score']
            pred.append(result)
        return pred


    def proba2craftScore(self,proba):
        result=[]
        for score in proba[:,1]:
            if (score>0.5-self.neu_range) and (score<0.5+self.neu_range):
                result.append({'label':'NEUTRAL','score':score})
                continue
            if score<0.5:
                result.append({'label':'NEGATIVE','score':score})
            else:
                result.append({'label':'POSITIVE','score':score})
        return result


    def predict(self,sentences):
        proba = self.predict_proba(sentences)
        ind = np.argmax(proba, axis=1)
        return np.array(self._classnames)[ind]


    def predict_proba(self,sentences):
        classifier = pipeline('sentiment-analysis')
        return np.array( [ [result['score'],1-result['score']] if result['label']=='NEGATIVE' else [1-result['score'],result['score']] \
              for result in classifier(sentences)])


class embeddings():

    def __init__(self,sent_model_name='roberta-base-nli-stsb-mean-tokens',token_model_name="bert-base-uncased"):
        self.sbert_model = SentenceTransformer(sent_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(token_model_name)
        self.bert_model = BertModel.from_pretrained(token_model_name, return_dict=True)


    def _get_bert_embeddings(self,string):
        inputs = self.tokenizer(string, return_tensors="pt")
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[0,1:-1,:].detach()


    def bert_emb(self,words):
        return np.array( [self._get_bert_embeddings(w).tolist()[0] for w in words])


    def sbert_emb(self,sentences):
        return self.sbert_model.encode(sentences, convert_to_tensor=True).numpy()


    def cos_sim(self,embedding1,embedding2):
        return util.pytorch_cos_sim(torch.from_numpy(embedding1),torch.from_numpy(embedding2))


class tokenizer():

    def __init__(self,model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def token(self,sentences):
        if isinstance(sentences,list):
            return [self.tokenizer.tokenize(sent) for sent in sentences]
        elif isinstance(sentences,str):
            return self.tokenizer.tokenize(sentences)
        else:
            return []