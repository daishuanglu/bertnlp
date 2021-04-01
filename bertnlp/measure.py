from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter


def plotConfMat(truth,pred,classname,fig_name='./confmat'):
    conf_mat=confusion_matrix(truth,pred,labels=classname)
    columns=classname
    df_cm = DataFrame(conf_mat, index=columns, columns=columns)
    plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
    plt.tight_layout()
    #plt.show()
    figure = ax.get_figure()
    figure.savefig(fig_name, dpi=400)
    plt.close()
    return conf_mat


def avg_prec_rec(pred,truth):
    classes=set(sum(truth,[]))
    Nsamples=len(truth)
    cls_tp={c:0 for c in classes}
    cls_num_pred={c:0 for c in classes}
    num_pred=0
    num_truth=0
    num_tp=0
    for i,p in enumerate(pred):
        tp_cls =set(p).intersection(set(truth[i]))
        tp=len(tp_cls)
        num_tp+=tp
        num_pred+=len(p)
        num_truth+=len(truth[i])
        for c,n in Counter(tp_cls).items():
            cls_tp[c]+=n
        for c in truth[i]:
            cls_num_pred[c]+=len(p)

    prec =num_tp/ num_pred
    rec = num_tp/ num_truth
    return {'precision':prec,'recall':rec,'N':Nsamples,'prec_by_class':{c:cls_tp[c]/cls_num_pred[c] for c in classes}}




def predict_func(data,model,conf_th):
    return model.predict(data,conf_thresh=conf_th)

def plotPrecisionRecall(test_data,truth,model,predict_func,conf_thresh_range=np.arange(0, 1, 0.1),fig_path='current_method'):
    # truth - a multilabeled groundtruth label list, categories are splitted by comma ","
    best_thresh=0.0
    best_prec_rec = {}
    best_prec = 0.0
    prec = []
    rec = []
    for conf_th in conf_thresh_range:
        feat_ftpred=predict_func(test_data,model,conf_th)

        prec_rec = avg_prec_rec(feat_ftpred, truth)
        if prec_rec['precision'] > best_prec:
            best_prec_rec = prec_rec
            best_prec = prec_rec['precision']
            best_thresh=conf_th
        prec.append(prec_rec['precision'])
        rec.append(prec_rec['recall'])

    ind = np.argsort(rec)
    prec = np.array(prec)[ind]
    rec = np.array(rec)[ind]

    plt.figure()
    plt.plot(rec, prec, 'orange')
    plt.xlabel('Recall')
    plt.ylabel("Precision")
    plt.title('Precision-Recall curve for feature detection.')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()
    return best_prec_rec,best_thresh