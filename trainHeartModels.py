from bert.text_classifier import knnbert,fasttextClf,trainer
from bert.pipeline import sentiment
import csv
import numpy as np
from bert.utils import clean_str,parse_heart_csv

from bert.fuzzy_matcher import feat_predict_func
from bert.measure import plotConfMat,plotPrecisionRecall,avg_prec_rec
from aspectnlp.aspect_detector import aspectDetector


negative_categories=['Critique','Issue','Enhancement Request']
toTrainCat=False
toTrainSubcat=False
toTrainFeatDet=True
sentiScorer = sentiment()


def drop_class(X,y,classname):
    sel_id=[i for i,yy in enumerate(y) if yy!=classname]
    return [X[i] for i in sel_id],[y[i] for i in sel_id]


data,senti_cat,subcat,senti_label,featureMent=parse_heart_csv('../heart_data/heart_train_SeptNovDec2020.csv','cp1252')
test_data,test_senti_cat,test_subcat,test_senti_label,test_featureMent=parse_heart_csv('../heart_data/heart_test_Aug2020.csv','cp1252')
sbert_model_name='roberta-base-nli-stsb-mean-tokens'

cat_data_tr,cat_tr=drop_class(data,senti_cat,"Critique")
cat_data_te, cat_te = drop_class(test_data, test_senti_cat, "Critique")
if toTrainCat:
    cat_model = knnbert(sbert_model_name=sbert_model_name)
    trainer(cat_data_tr+cat_data_te,cat_tr+cat_te,cat_model,eval_round=10)
    cat_model =trainer(cat_data_tr+cat_data_te,cat_tr+cat_te,cat_model,save_model_path='../ext_models/heartSentiCat.pkl')
else:
    cat_model = knnbert(sbert_model_name=sbert_model_name,pretrained='../ext_models/heartSentiCat.pkl')
    #senti_cat_model=torch.load('./ext_models/heartSentiCat.pkl')
cat_pred=cat_model.predict(cat_data_te)
print('senti_cat sbert model test accuracy {}'.format((cat_pred==np.array(cat_te)).mean() ))
plotConfMat(cat_te,cat_pred,cat_model.classes_,'cat_sbertClf_confmat.png')


subcat_data_tr,subcat_tr=drop_class(data[:len(subcat)],subcat,"Other")
subcat_data_te, subcat_te = drop_class(test_data, test_subcat, "Other")
if toTrainSubcat:
    subcat_model = knnbert(sbert_model_name=sbert_model_name)
    trainer(subcat_data_tr+subcat_data_te,subcat_tr+subcat_te,subcat_model,eval_round=10,test_size=0.2)
    subcat_model =trainer(subcat_data_tr+subcat_data_te,subcat_tr+subcat_te,subcat_model,save_model_path='../ext_models/heartSubCat.pkl')
else:
    subcat_model = knnbert(sbert_model_name=sbert_model_name, pretrained='../ext_models/heartSubCat.pkl')
subcat_pred=subcat_model.predict(subcat_data_te)
print('subcat sbert model test accuracy {}'.format((subcat_pred==np.array(subcat_te)).mean() ))
plotConfMat(subcat_te,subcat_pred,subcat_model.classes_,'subcat_sbertClf_confmat.png')


'''
#test_senti_cat=['Issue' if c=='Critique' else c for c in test_senti_cat]
test_senti_pred,test_senti_prob=list(zip(*[(pred['label'].lower(),pred['score']) for pred in sentiScorer.score(test_data)]))
senti_acc=np.array([test_senti_pred[i]==l.lower() for i,l in enumerate(test_senti_label)]).mean()
print('sentiment model accuracy {}'.format(senti_acc))
'''

featurelist=list(set(sum(test_featureMent,[])+ sum(featureMent,[])))

asp_det=aspectDetector('../ext_models/custom_emb.vec.bin')
tr_data=[data[i] for i,f in enumerate(featureMent) if 'None' not in ' '.join(f)]
asp_str=[' '.join(i['aspect']) for i in asp_det.detect(tr_data)]
test_asp_str=[' '.join(i['aspect']) for i in asp_det.detect(test_data)]

X_tr=asp_str
X_te=test_asp_str

if toTrainFeatDet:
    featMent_ftmodel=fasttextClf()
    tr_featMent=[''.join(f) for f in featureMent if 'None' not in ' '.join(f)]
    featMent_ftmodel.fit(X_tr,tr_featMent,lr=1.0,epoch=100,wordNgrams=2,loss='ova')
    featMent_ftmodel.model.save_model('../ext_models/featMent_ftClf.bin')
else:
    featMent_ftmodel = fasttextClf(pretrained='../ext_models/featMent_ftClf.bin')


combinedModel={'ftmodel':featMent_ftmodel,'featlist':featurelist}


def predict_func(data,model,conf_th):
    return feat_predict_func(data,model,conf_th)[0]

feat_pred,_=feat_predict_func(X_te,combinedModel,0.2)
print(feat_pred)
test_featMent=[''.join(f) for f in test_featureMent]
print(test_featureMent)

best_prec_rec,best_thresh=plotPrecisionRecall(X_te,test_featureMent,combinedModel, predict_func, \
                                              conf_thresh_range=np.arange(0,1,0.1),fig_path= './aspect_ftClf_prec_rec.png')

print('The best Precision- Recall is evaluated at confident threshold {}:'.format(best_thresh))
print('Precision: ',best_prec_rec['precision'],'Recall: ',best_prec_rec['recall'])
for c,rate in best_prec_rec['prec_by_class'].items():
    print('class',c,': ',rate)