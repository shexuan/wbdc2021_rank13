import pandas as pd
import numpy as np
import os, gc, sys, tqdm
sys.path.append('../')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from sklearn.preprocessing import Normalizer
from sklearn import metrics
from pprint import pprint
from sklearn.cluster import KMeans, MiniBatchKMeans

from utils import reduce_mem_usage, dict2model
plt.rc('font', family='SimHei')
plt.rcParams['font.sans-serif'] = ['SimHei'] 

plt.style.use('ggplot')

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
pd.set_option('precision', 5) #设置精度
pd.set_option('display.float_format', lambda x: '%.5f' % x) #为了直观的显示数字，不采用科学计数法

# jupyter notebook中设置交互式输出
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', 100)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


# missingno  缺失处理库
import missingno

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

RAW_DATA_PATH = '../../wbdc2021/data/wedata/wechat_algo_data2/'
OUT_PATH = '../my_data/'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

if not os.path.exists(f'{OUT_PATH}/feedid_text_features/'):
    os.makedirs(f'{OUT_PATH}/feedid_text_features/')


MACHINE_TAG_PROB = 0.3
TFIDF_MAXDF = 0.7
TFIDF_MINDF = 5
TFIDF_NGRAM = (1, 2)


#####################################################################################
##################### 1、预处理 ######################################################
def jcd(manu, machine):
    manu = set(manu.split(';'))
    machine = set(machine.split(';'))
    return len(set(manu)&set(machine))/len(set(manu)|set(machine))  

def set_bins(x):
    if x<=15:
        return 0
    elif x<=30:
        return 1
    elif x<=60:
        return 2
    elif x<=300:
        return 3
    else:
        return 4

feed_info = pd.read_csv(f'{RAW_DATA_PATH}/feed_info.csv', header=0)
feed_info.shape
feed_info.head(2)

# 1.1 处理feedinfo中的machine_tag_list字段，去掉低概率的tag，这里概率阈值取0.3
feed_info['machine_tag_list'] = feed_info['machine_tag_list']\
    .apply(lambda x: ';'.join(i.split()[0] for i in x.split(';') if float(i.split()[1])>MACHINE_TAG_PROB) 
           if isinstance(x, str) else x)

# 1.2 生成manual_tag 和 machine_tag 相关性字段
feed_info.loc[~((feed_info.manual_tag_list.isnull())|(feed_info.manual_tag_list=='')|\
              (feed_info.machine_tag_list.isnull())|(feed_info.machine_tag_list=='')), 'tag_manu_machine_corr'] = \
    feed_info.loc[~((feed_info.manual_tag_list.isnull())|(feed_info.manual_tag_list=='')|\
              (feed_info.machine_tag_list.isnull())|(feed_info.machine_tag_list=='')), ['manual_tag_list','machine_tag_list']]\
    .apply(lambda row: jcd(row.manual_tag_list, row.machine_tag_list),axis=1)
feed_info.loc[feed_info['tag_manu_machine_corr'].isna(), 'tag_manu_machine_corr'] = feed_info['tag_manu_machine_corr'].mean()


# 1.3 填补缺失. 由于机器标注的人工标注的词表是相同的，因此可以互相用来填充缺失
#feed_info.loc[feed_info.manual_keyword_list.isnull(), 'manual_keyword_list'] = feed_info['machine_keyword_list']
#feed_info.loc[feed_info.machine_keyword_list.isnull(), 'machine_keyword_list'] = feed_info['manual_keyword_list']

feed_info.loc[(feed_info.manual_tag_list.isnull())|(feed_info.manual_tag_list==''),'manual_tag_list'] = \
    feed_info.loc[(feed_info.manual_tag_list.isnull())|(feed_info.manual_tag_list==''), 'machine_tag_list']

feed_info.loc[(feed_info.machine_tag_list.isnull())|(feed_info.machine_tag_list==''),'machine_tag_list'] = \
    feed_info.loc[(feed_info.machine_tag_list.isnull())|(feed_info.machine_tag_list==''), 'manual_tag_list']


# 1.4 bgm缺失填充. 使用authorid进行填充，且新建一个特征表示是否缺失
feed_info['bgm_na'] = 0
feed_info.loc[feed_info.bgm_singer_id.isnull(), 'bgm_na'] = 1
feed_info.loc[feed_info.bgm_song_id.isna(), 'bgm_song_id'] = \
    (feed_info.loc[feed_info.bgm_song_id.isna(), 'authorid']*-1)

feed_info.loc[feed_info.bgm_singer_id.isna(), 'bgm_singer_id'] = \
    (feed_info.loc[feed_info.bgm_singer_id.isna(), 'authorid']*-1)


# 1.5 处理videoplayseconds
# 离散分箱
feed_info['videoplayseconds_bin'] = feed_info['videoplayseconds'].apply(set_bins)
# log transform
feed_info['videoplayseconds'] = np.log(feed_info['videoplayseconds']+1)
# 对feed_info中的videoplayseconds进行归一化，对'bgm_song_id', 'bgm_singer_id'进行缺失值填充
mms = MinMaxScaler(feature_range=(0, 1))
feed_info['videoplayseconds'] = mms.fit_transform(feed_info[['videoplayseconds']])


#####################################################################################
##################### 2、生成 TfIDF 特征 ##############################################

def get_tfidf(x, n_components=16, col_prefix='manu_tag_tfidf_'):
    """对于视频的序列字段使用tfidf提取特征并生成主题
    """
    vectorizer = TfidfVectorizer(max_df=TFIDF_MAXDF,
                                 min_df=TFIDF_MINDF,  
                                 ngram_range=TFIDF_NGRAM,
                                 use_idf=True,
                                 norm='l2')
    
    tfidf_vec = vectorizer.fit_transform(x)
    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
    tfidf_features = svd.fit_transform(tfidf_vec)
    
    df_tfidf = pd.DataFrame(tfidf_features)
    df_tfidf.columns = [col_prefix+str(i) for i in range(n_components)]
    
    return df_tfidf, tfidf_vec


def get_topics(x, n_topics=10, col_prefix='manu_tag_topic_'):
    """生成每个feed所属于的主题
    """
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='batch', 
                                    max_iter=10, random_state=123)
    lda_topics = lda.fit_transform(x)
    df_topics = pd.DataFrame(lda_topics)
    df_topics.columns = [col_prefix+str(i) for i in range(n_topics)]
    
    df_topics[col_prefix+'class'] = np.argmax(lda_topics,axis=1)

    return df_topics


def get_text_features(df, colname, n_components=16, n_topics=10, col_prefix=None, idcol='feedid'):
    df = df[(~df[colname].isnull())&(df[colname]!='')][[idcol, colname]]
    df[colname] = df[colname].apply(lambda x:' '.join(x.split(';')))
    
    df_tfidf, tfidf_vec =  get_tfidf(df[colname], n_components, col_prefix+'_tfidf_')    
    df_topics = get_topics(tfidf_vec, n_topics, col_prefix+'_topic_')
    
    df = pd.concat([df, df_tfidf, df_topics], axis=1)
    
    return df


# 2.1 处理 feed 文本特征
# 生成 feed manu_tag, machine_tag, manual_keyword_list, machine_keyword_list, decription的 tfidf 和 LDA 特征
df_manu_tag = get_text_features(feed_info, 'manual_tag_list', col_prefix='feed_manu_tag')
df_machine_tag = get_text_features(feed_info, 'machine_tag_list', col_prefix='feed_machine_tag')
df_manu_kw = get_text_features(feed_info, 'manual_keyword_list', col_prefix='feed_manu_kw')
df_machine_kw = get_text_features(feed_info, 'machine_keyword_list', col_prefix='feed_machine_kw')
df_desc = get_text_features(feed_info, 'description', col_prefix='feed_description')

# df_feed = feed_info[['feedid','authorid','videoplayseconds','bgm_song_id','bgm_singer_id']]\
#             .merge(df_manu_tag, on='feedid', how='left')\
#             .merge(df_machine_tag, on='feedid', how='left')\
#             .merge(df_manu_kw, on='feedid', how='left')\
#             .merge(df_machine_kw, on='feedid', how='left')\
#             .merge(df_desc, on='feedid', how='left')\
#             .drop(columns=['description','machine_keyword_list','manual_keyword_list',
#                           'machine_tag_list','manual_tag_list'])

# df_feed.to_pickle('wechat_algo_data1/my_data/feedid_text_features/feed_text_features.pkl')


# 2.2 处理 author 文本特征
# 将同一个author多个feed的manu_tag, machine_tag, manu_kw, machine_kw以及desc tag进行聚合，合并文本，然后提取文本信息

def mysplit(x, sep=' '):
    lst = []
    for i in x:
        for j in i:
            lst.extend(j.split(sep))
    return ' '.join(lst)

def get_author_text_features(df, colname, idcol='authorid', sep=' ', n_components=16, n_topics=10, col_prefix=None):
    """先对author的对个feed信息进行聚合合并所有feed的文本信息
    """
    df = df[~df[colname].isnull()][[idcol, colname]]
    df = df.groupby(idcol).agg(lambda x: list(x)).apply(lambda x: mysplit(x, sep=sep), axis=1).reset_index()
    df.columns = [idcol, colname]

    # 提取文本信息
    df_tfidf, tfidf_vec =  get_tfidf(df[colname], n_components, col_prefix+'_tfidf_')    
    df_topics = get_topics(tfidf_vec, n_topics, col_prefix+'_topic_')
    
    df = pd.concat([df, df_tfidf, df_topics], axis=1)

    return df

author_desc = get_author_text_features(feed_info, 'description', idcol='authorid', 
                                       sep=' ', col_prefix='author_description')
author_manu_kw = get_author_text_features(feed_info, 'manual_keyword_list', 
                                          idcol='authorid', sep=';', col_prefix='author_manu_kw')
author_machine_kw = get_author_text_features(feed_info, 'machine_keyword_list', 
                                             idcol='authorid', sep=';', col_prefix='author_machine_kw')
author_manu_tag = get_author_text_features(feed_info, 'manual_tag_list', idcol='authorid', 
                                           sep=';', col_prefix='author_manu_tag')
author_machine_tag = get_author_text_features(feed_info, 'machine_tag_list', idcol='authorid', 
                                              sep=';', col_prefix='author_machine_tag')

df_feed = feed_info[['feedid','authorid','bgm_song_id','bgm_singer_id',
                     'videoplayseconds_bin','bgm_na','videoplayseconds','tag_manu_machine_corr']]\
            .merge(df_manu_tag, on='feedid', how='left')\
            .merge(df_machine_tag, on='feedid', how='left')\
            .merge(df_manu_kw, on='feedid', how='left')\
            .merge(df_machine_kw, on='feedid', how='left')\
            .merge(df_desc, on='feedid', how='left')\
            .merge(author_desc.drop(columns=['description']), on='authorid', how='left')\
            .merge(author_manu_kw.drop(columns=['manual_keyword_list']), on='authorid', how='left')\
            .merge(author_machine_kw.drop(columns=['machine_keyword_list']), on='authorid', how='left')\
            .merge(author_manu_tag.drop(columns=['manual_tag_list']), on='authorid', how='left')\
            .merge(author_machine_tag.drop(columns=['machine_tag_list']), on='authorid', how='left')

# df_feed = reduce_mem_usage(df_feed)

df_feed.to_pickle(f'{OUT_PATH}/feedid_text_features/feed_author_text_features.pkl')


#####################################################################################
##################### 3、填充部分缺失的 TfIDF 特征 #####################################

# 根据author来进行缺失填充，依据是：同一个author发布的视频具有较大的相似度，因此若同一个author发布的多个视频中有一个是缺失的，
# 可用其他的多个视频的均值来进行填充；然而，author也有全部feed缺失的，author缺失的暂时不管。（也可以考虑利用author相似性，
# 寻找相似性author，然后用相似性author的值来进行填充）

def fillna_tfidf_by_author(df, field_cols, idcol='authorid'):
    """根据author来进行缺失填充,即同一个author发布的不同feed若有内容缺失，则使用此author其他feed的均值来进行填充
    """
    # 需要先将全零列设置为Nan，后面一起填充
    df.loc[(df[field_cols]==0).all(axis=1), field_cols] = np.nan
    print('fillna before not na rows: ', df.query(f'{field_cols[0]}=={field_cols[0]}').shape)
    df[field_cols] = df[[idcol]+field_cols].groupby(idcol).transform(lambda x: x.fillna(x.mean()))
    print('fillna after not na rows: ', df.query(f'{field_cols[0]}=={field_cols[0]}').shape)
    return df

df_feed = pd.read_pickle(f'{OUT_PATH}/feedid_text_features/feed_author_text_features.pkl')

feed_manu_tag_cols = ['feed_manu_tag_tfidf_'+str(i) for i in range(16)]
feed_machine_tag_cols = ['feed_machine_tag_tfidf_'+str(i) for i in range(16)]
feed_manu_kw_cols = ['feed_manu_kw_tfidf_'+str(i) for i in range(16)]
feed_machine_kw_cols = ['feed_machine_kw_tfidf_'+str(i) for i in range(16)]
feed_desc_cols = ['feed_description_tfidf_'+str(i) for i in range(16)]

author_manu_tag_cols = ['author_manu_tag_tfidf_'+str(i) for i in range(16)]
author_machine_tag_cols = ['author_machine_tag_tfidf_'+str(i) for i in range(16)]
author_manu_kw_cols = ['author_manu_kw_tfidf_'+str(i) for i in range(16)]
author_machine_kw_cols = ['author_machine_kw_tfidf_'+str(i) for i in range(16)]
author_desc_cols = ['author_description_tfidf_'+str(i) for i in range(16)]


# 3.1 处理全零tfidf向量.
# tfidf向量中有很多feedid的manu_tag, machine_tag的向量全为0，将这些向量值填充为缺失先.
# 后续待进一步研究处理这些全零值
df_feed.loc[(df_feed[feed_manu_tag_cols]==0).all(axis=1), feed_manu_tag_cols] = np.nan
df_feed.loc[(df_feed[feed_machine_tag_cols]==0).all(axis=1), feed_machine_tag_cols] = np.nan
df_feed.loc[(df_feed[feed_manu_kw_cols]==0).all(axis=1), feed_manu_kw_cols] = np.nan
df_feed.loc[(df_feed[feed_machine_kw_cols]==0).all(axis=1), feed_machine_kw_cols] = np.nan
df_feed.loc[(df_feed[feed_desc_cols]==0).all(axis=1), feed_desc_cols] = np.nan

df_feed.loc[(df_feed[author_manu_tag_cols]==0).all(axis=1), author_manu_tag_cols] = np.nan
df_feed.loc[(df_feed[author_machine_tag_cols]==0).all(axis=1), author_machine_tag_cols] = np.nan
df_feed.loc[(df_feed[author_manu_kw_cols]==0).all(axis=1), author_manu_kw_cols] = np.nan
df_feed.loc[(df_feed[author_machine_kw_cols]==0).all(axis=1), author_machine_kw_cols] = np.nan
df_feed.loc[(df_feed[author_desc_cols]==0).all(axis=1), author_desc_cols] = np.nan

# 3.2 根据author mean来填充缺失值
df_feed = fillna_tfidf_by_author(df_feed, feed_desc_cols, idcol='authorid')
df_feed = fillna_tfidf_by_author(df_feed, feed_manu_tag_cols, idcol='authorid')
df_feed = fillna_tfidf_by_author(df_feed, feed_machine_tag_cols, idcol='authorid')
df_feed = fillna_tfidf_by_author(df_feed, feed_machine_kw_cols, idcol='authorid')
df_feed = fillna_tfidf_by_author(df_feed, feed_manu_kw_cols, idcol='authorid')

# 3.3 意义相近字段相似性填充，即 manual 和 machine 结果互补缺失值，
# 注：此步有误，不能在此处互补缺失，因为此处是已经经过处理的tfidf特征，互补缺失只能在原始数据处进行互相填补
# df_feed.loc[(df_feed[feed_manu_tag_cols[0]].isnull()) & (~df_feed[feed_machine_tag_cols[0]].isnull()), feed_manu_tag_cols] = \
#     df_feed.loc[(df_feed[feed_manu_tag_cols[0]].isnull()) & (~df_feed[feed_machine_tag_cols[0]].isnull()), feed_machine_tag_cols].values

# df_feed.loc[(df_feed[feed_machine_tag_cols[0]].isnull()) & (~df_feed[feed_manu_tag_cols[0]].isnull()), feed_machine_tag_cols] = \
#     df_feed.loc[(df_feed[feed_machine_tag_cols[0]].isnull()) & (~df_feed[feed_manu_tag_cols[0]].isnull()), feed_manu_tag_cols].values

# # 意义相近字段相似性填充
# df_feed.loc[(df_feed[feed_manu_kw_cols[0]].isnull()) & (~df_feed[feed_machine_kw_cols[0]].isnull()), feed_manu_kw_cols] = \
#     df_feed.loc[(df_feed[feed_manu_kw_cols[0]].isnull()) & (~df_feed[feed_machine_kw_cols[0]].isnull()), feed_machine_kw_cols].values

# df_feed.loc[(df_feed[feed_machine_kw_cols[0]].isnull()) & (~df_feed[feed_manu_kw_cols[0]].isnull()), feed_machine_kw_cols] = \
#     df_feed.loc[(df_feed[feed_machine_kw_cols[0]].isnull()) & (~df_feed[feed_manu_kw_cols[0]].isnull()), feed_manu_kw_cols].values

df_feed.to_pickle(f'{OUT_PATH}feedid_text_features/feed_author_text_features_fillna_by_author_rm0.pkl')



#####################################################################################
##################### 4、PCA压缩官方feed embedding ###################################

# 4.1 处理一下embedding，转为dataframe
feed_emb = pd.read_csv(f'{RAW_DATA_PATH}/feed_embeddings.csv', header=0)
feed_emb.shape

feed_emb_dic = {}
for idx, feed in tqdm.tqdm(feed_emb.iterrows()):
    feed_emb_dic[feed['feedid']] = np.array([float(i) for i in feed['feed_embedding'].split()])

df = pd.DataFrame(feed_emb_dic).T
df.reset_index(inplace=True)

df.columns = ['feedid']+['feed_emb_'+str(i) for i in range(512)]
df.to_pickle(f'{OUT_PATH}/feed_embedding_d512.pkl')

# 存为gensim word2vec model格式
raw_mm = dict2model(df, df.columns[0], df.columns[1:], 
                    f'{OUT_PATH}/official_feed_emb.d512.pkl')

# 4.2 pca降维
df.index = df['feedid']
df.drop(columns=['feedid'], inplace=True)
pca = PCA(n_components=32, whiten=True)
new_emb = pca.fit_transform(df.values)
df_new_emb = pd.DataFrame(new_emb)
df_new_emb.index = df.index
df_new_emb.reset_index(inplace=True)

df_new_emb.columns = ['feedid'] + ['feed_emb_'+str(i) for i in range(32)]
df_new_emb.to_pickle(f'{OUT_PATH}/feed_embedding_pca_d32.pkl')

# 存为gensim word2vec model格式
raw_mm = dict2model(df_new_emb, df_new_emb.columns[0], df_new_emb.columns[1:], 
                    f'{OUT_PATH}/official_feed_emb_pca.d32.pkl')

#####################################################################################
##################### 5、提取tag、keyword文本，并将其padding对齐 ########################
def word2index(df, col='manual_tag_list', max_len=11):
    df = df[['feedid',col]].explode(col)
    lbe = LabelEncoder()
    df[col] = lbe.fit_transform(df[col])+1
    print(f'unique {col} word numbers: ', len(lbe.classes_))
    df = df.groupby('feedid').agg(list).reset_index()
    df[col] = pad_sequences(list(df[col]), maxlen=max_len, padding='post').tolist()
    return df


feed = pd.read_pickle(f'{OUT_PATH}feedid_text_features/feed_author_text_features_fillna_by_author_rm0.pkl')
feed1 = feed[['feedid','manual_tag_list', 'machine_tag_list', 'manual_keyword_list', 'machine_keyword_list']]
# 填补缺失
feed1.loc[(feed1.manual_tag_list.isna()) | (feed1.manual_tag_list==''), 'manual_tag_list'] = \
    (feed1.loc[(feed1.manual_tag_list.isna()) | (feed1.manual_tag_list==''), 'feedid']*-1).astype('str') 
feed1.loc[(feed1.machine_tag_list.isna()) | (feed1.machine_tag_list==''), 'machine_tag_list'] = \
    (feed1.loc[(feed1.machine_tag_list.isna()) | (feed1.machine_tag_list==''), 'feedid']*-1).astype('str') 
feed1.loc[feed1.manual_keyword_list.isna(), 'manual_keyword_list'] = (feed1.loc[feed1.manual_keyword_list.isna(), 'feedid']*-1).astype('str') 
feed1.loc[feed1.machine_keyword_list.isna(), 'machine_keyword_list'] = (feed1.loc[feed1.machine_keyword_list.isna(), 'feedid']*-1).astype('str') 

feed1[['manual_tag_list','machine_tag_list','manual_keyword_list','machine_keyword_list']] = \
    feed1[['manual_tag_list','machine_tag_list','manual_keyword_list','machine_keyword_list']].applymap(lambda x:x.split())

df_manu_tag = word2index(feed1, col='manual_tag_list', max_len=11)
df_machine_tag = word2index(feed1, col='machine_tag_list', max_len=11)
df_manu_kw = word2index(feed1, col='manual_keyword_list', max_len=18)
df_machine_kw = word2index(feed1, col='machine_keyword_list', max_len=16)

# unique manual_tag_list word numbers:  533
# unique machine_tag_list word numbers:  495
# unique manual_keyword_list word numbers:  58486
# unique machine_keyword_list word numbers:  24416

df_padded = df_manu_tag.merge(df_machine_tag, on='feedid', how='inner')\
                .merge(df_manu_kw, on='feedid', how='inner')\
                .merge(df_machine_kw, on='feedid', how='inner')
df_padded.to_pickle(f'{OUT_PATH}/feedid_text_features/feed_tag_kw_padded_text.pkl')
