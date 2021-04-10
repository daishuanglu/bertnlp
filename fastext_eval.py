from bertnlp.text_classifier import fasttextClf
from bertnlp.utils import clean_str
import numpy as np
from bertnlp.measure import plotPrecisionRecall,predict_func
import csv
import os
import torch


class dataloader():

    def __init__(self,fpath,code='utf-8'):

        self.fpath=fpath
        self.encoding=code
        self._reset()
        self.nsamples=0
        for l in self.f:
            self.nsamples+=1

        self._reset()
        return

    def batch(self,bsize=1024):
        ind= sorted( np.random.randint(self.nsamples,size=bsize))

        data,label=[],[]
        for i,sample in enumerate( self.f):
            if i in ind:
                data.append(sample['Text'])
                label.append(sample['Label'].split('|'))

        self._reset()
        return data,label

    def full(self):

        data, label=[],[]
        for sample in self.f:
            data.append(sample['Text'])
            label.append(sample['Label'].split('|'))
        self._reset()
        return data,label

    def _reset(self):
        self.f= csv.DictReader(open(self.fpath, 'r', encoding=self.encoding))
        return


def pytorch_train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def pytorch_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


datadir='movie'
encoding='utf-8'
trainLoader= dataloader(os.path.join(datadir,'train.csv'),code=encoding)
valLoader=dataloader(os.path.join(datadir,'val.csv'),code=encoding)


train_data,train_label= trainLoader.full()
val_data,val_label=valLoader.full()
train_data+=val_data
train_label+=val_label

train_data=[clean_str(s) for s in train_data]
val_data=[clean_str(s) for s in val_data]

ftmodel=fasttextClf()
ftmodel.fit(train_data,train_label,lr=1.0,epoch=120,wordNgrams=2,loss='ova')
#ftmodel.model.save_model('./featMent_ftClf.bin')


testLoader=dataloader(os.path.join(datadir,'test.csv'),code=encoding)
test_data,test_label=testLoader.full()
test_data=[clean_str(s) for s in test_data]

best_prec_rec,best_thresh=plotPrecisionRecall(test_data,test_label,ftmodel, predict_func, \
                                              conf_thresh_range=np.arange(0,1,0.1),fig_path= './ftClf_prec_rec.png')

print('The best Precision- Recall is evaluated at confident threshold {}:'.format(best_thresh))
print('Precision: ',best_prec_rec['precision'],', Recall: ',best_prec_rec['recall'])
print('Avg. Hamming loss: ',best_prec_rec['avg_hamming_loss'],', Micro-F1: ',best_prec_rec['f1'])
for c,rate in best_prec_rec['prec_by_class'].items():
    print('class',c,': ',rate)