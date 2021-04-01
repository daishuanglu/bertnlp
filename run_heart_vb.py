from bert.vb_service import bertService
from bert.utils import parseCsv

data_path_prefix='../HEART_intelligence/heart_data/'
rowIds,corpus=parseCsv(data_path_prefix+'heart_test_Aug2020.csv')

serv=bertService('./','')
path_prefix='../HEART_intelligence/heart_ext_models/'
serv.serv_heart(rowIds,corpus,embedding_path=path_prefix+'custom_emb.vec.bin',sentiCat_model_path=path_prefix+'heartSentiCat.pkl',\
                featMent_model_path=path_prefix+'featMent_ftClf.bin',\
subCat_model_path=path_prefix+'heartSubCat.pkl',chunksize=16,matcher_threshold=0.2,neu_range=0.15)
