from sklearn.metrics import confusion_matrix,hamming_loss
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


def seq_hamming_loss(seq,truth_seq):
    lseq=len(seq)
    ltruth=len(truth_seq)
    if lseq<ltruth:
        if isinstance(truth_seq[0],int) or isinstance(truth_seq[0],bool) or isinstance(truth_seq[0],float):
            seq+=[-10000 for _ in range(ltruth-lseq)]
        elif isinstance(truth_seq[0],str):
            seq+=['' for _ in range(ltruth-lseq)]
    elif lseq>ltruth:
        seq=seq[:ltruth]

    return hamming_loss(truth_seq,seq)


def avg_prec_rec(pred,truth,ordered=False):
    classes=set(sum(truth,[]))
    Nsamples=len(truth)
    cls_tp={c:0 for c in classes}
    cls_num_pred={c:0 for c in classes}
    num_pred=0
    num_truth=0
    num_tp=0
    num_fp=0
    num_fn=0
    hamming_loss=0.0
    for i,p in enumerate(pred):
        tp_cls =set(p).intersection(set(truth[i]))
        if ordered:
            hamming_loss +=seq_hamming_loss(list(p),truth[i])
        else:
            ltruth_i = len(truth[i])
            hamming_loss += len(set(p[:ltruth_i]).intersection(set(truth[i]))) / ltruth_i
        tp=len(tp_cls)
        num_tp+=tp
        fp = len(p) - tp
        num_fp+=fp
        fn=len(set(truth[i])-set(p))
        num_fn+=fn
        num_pred+=len(p)
        num_truth+=len(truth[i])
        for c,n in Counter(tp_cls).items():
            cls_tp[c]+=n
        for c in truth[i]:
            cls_num_pred[c]+=len(p)

    prec =num_tp/ num_pred
    rec = num_tp/ num_truth
    f1score=num_tp/(num_tp+0.5*(num_fp+num_fn))

    return {'avg_hamming_loss':hamming_loss/Nsamples,'f1':f1score,'precision':prec,'recall':rec,'num_of_samples':Nsamples,'prec_by_class':{c:cls_tp[c]/cls_num_pred[c] for c in classes}}


def predict_func(data,model,conf_th):
    return model.predict(data,conf_thresh=conf_th)


def plotPrecisionRecall(test_data,truth,model,predict_func,conf_thresh_range=np.arange(0, 1, 0.1),fig_path='current_method',ordered=False):
    # truth - a multilabeled groundtruth label list, categories are splitted by comma ","
    best_thresh=0.0
    best_prec_rec = {}
    best_f1 = 0.0
    prec = []
    rec = []
    for conf_th in conf_thresh_range:
        feat_ftpred=predict_func(test_data,model,conf_th)

        prec_rec = avg_prec_rec(feat_ftpred, truth,ordered)
        if prec_rec['f1'] > best_f1:
            best_prec_rec = prec_rec
            best_f1 = prec_rec['f1']
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
    plt.title('Precision-Recall curve for multilabel detection.')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()
    return best_prec_rec,best_thresh