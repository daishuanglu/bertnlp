import csv

headers=['Label','Text']


def preproc_movie():
    col2header={'plot_synopsis':'Text','tags':'Label'}

    f = csv.DictReader(open('./movie/mpst_full_data.csv', 'r', encoding='utf-8'))
    ftr = csv.DictWriter(open('./movie/train.csv', 'w', encoding='utf-8', newline=''), fieldnames=headers)
    ftr.writeheader()
    fval = csv.DictWriter(open('./movie/val.csv', 'w', encoding='utf-8', newline=''), fieldnames=headers)
    fval.writeheader()
    fte = csv.DictWriter(open('./movie/test.csv', 'w', encoding='utf-8', newline=''), fieldnames=headers)
    fte.writeheader()
    for ii,line in enumerate(f):
        if line['split'] in ['train','test','val']:
            rowDict={}
            for k in col2header.keys():
                if k == 'tags':
                    rowDict[col2header[k]] = '|'.join([w.strip().replace(' ','_') for w in line[k].split(',')])
                else:
                    rowDict[col2header[k]]=line[k]
        else:
            return
        if line['split']=='train':
            ftr.writerow(rowDict)
        if line['split']=='test':
            fte.writerow(rowDict)
        if line['split']=='val':
            fval.writerow(rowDict)
        print(ii,' lines processed.')
    return

def main():
    preproc_movie()

    return



if __name__=='__main__':
    main()