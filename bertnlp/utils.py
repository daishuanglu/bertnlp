import re
import csv
import os

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def keep_alphanumeric(s):
    return re.sub(r"[^A-Za-z0-9 .']+", ' ', s)


subcatMap={"Access":"Miscellaneous","Early-Survey-Prompt":"Miscellaneous",\
           "Ease-of-Use":"FrontEnd","Performance":"BackEnd","Other":"Other",\
           "Reliability/Stability":"BackEnd","Trust/Data-Quality":"BackEnd","UI":"FrontEnd"}


def parse_csv(file,code='utf-8'):
    data=[]
    cat_label=[]
    subcat_label=[]
    senti_label=[]
    features=[]
    for sample in csv.DictReader(open(file,'r',encoding=code)):
        if len(sample['Text'])==0: continue
        #data.append(clean_str(sample['Text']))
        data.append(clean_str(sample['Text']))
        cat_label.append('-'.join( sample['Category'].split()))
        if len(sample['SubCategory'])>0:
            subcat='-'.join( sample['SubCategory'].split(',')[0].split())
            subcat_label.append(subcatMap[subcat])
        if 'Sentiment' in sample.keys():
            senti_label.append(sample['Sentiment'])
        if len(sample['FeatureMentioned'])>0:
            ff=[f.strip() for f in sample['FeatureMentioned'].split(',')]
            features.append(ff)
        else:
            features.append(['None'])
    return data,cat_label,subcat_label,senti_label,features


def get_example_data(split='train',code='utf-8'):
    import bertnlp
    data_dir = os.path.join(os.path.dirname(bertnlp.__file__), 'data')
    if split=='test':
        fpath = os.path.join(data_dir, 'heart_test_Aug2020.csv')
    else:
        fpath=os.path.join(data_dir, 'heart_train_SeptNovDec2020.csv')
    return parse_csv(fpath,code)
