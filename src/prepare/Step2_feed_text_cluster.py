import pandas as pd
import numpy as np
import os, gc, sys
sys.path.append('../')
import matplotlib.pyplot as plt
from utils import reduce_mem_usage, uAUC,compute_weighted_score

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans
import multiprocessing as mp
from functools import wraps
import time
from collections import Counter
from datetime import datetime
import seaborn as sns
import tqdm

import warnings
warnings.filterwarnings("ignore")

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

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


DATA_ROOT = '../my_data/'
find_best_K = False
#####################################################################################
##################### 1、寻找 feed、author 最佳聚类数量 ##############################################
# 利用前面生成的 tfidf 特征对 feed、author 进行聚类
# 注：这一步非常耗时

def run_kmeans(arr, k_min=2, k_max=33):
    """寻找最佳聚类数
    """
    calinskis = []
    inertias = []
    for n in tqdm.tqdm(range(k_min, k_max)):
        cluster = KMeans(n_clusters=n, random_state=123)
        pred = cluster.fit_predict(arr)
        
        calinski = metrics.calinski_harabaz_score(arr, pred)
        calinskis.append(calinski)
        inertia = cluster.inertia_
        inertias.append(inertia)
        
#         print("cluster numbers:", n)
#         print("calinski metrics:", calinski)
#         print("inertia_:", cluster.inertia_)

#         print('\n')
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    res = pd.DataFrame({'K':np.arange(k_min,k_max), 'calinski':calinskis, 'inertia':inertias})
    res.set_index('K').plot(kind='bar', secondary_y='inertia', figsize=(15,8))
    

if find_best_K:
    # 1.1 feed聚类. 寻找各个feed text tfidf特征的最佳聚类数量
    df = pd.read_pickle(f'{DATA_ROOT}/feedid_text_features/feed_author_text_features_fillna_by_author_rm0.pkl')

    desc_tfidf_cols = [c for c in df.columns if c.startswith('feed_description_tfidf_')]
    feed_desc = df[~df['feed_description_tfidf_0'].isnull()][desc_tfidf_cols]
    run_kmeans(feed_desc.values) 

    manu_tag_tfidf_cols = [c for c in df.columns if c.startswith('feed_manu_tag_tfidf_')]
    df_manu_tag = df[~df['feed_manu_tag_tfidf_0'].isnull()]
    run_kmeans(df_manu_tag[manu_tag_tfidf_cols].values) 

    machine_tag_tfidf_cols = [c for c in df.columns if c.startswith('feed_machine_tag_tfidf_')]
    df_machine_tag = df[~df['feed_machine_tag_tfidf_0'].isnull()]
    run_kmeans(df_machine_tag[machine_tag_tfidf_cols].values) 

    manu_kw_tfidf_cols = [c for c in df.columns if c.startswith('feed_manu_kw_tfidf_')]
    df_manu_kw = df[~df['feed_manu_kw_tfidf_0'].isnull()]
    run_kmeans(df_manu_kw[manu_kw_tfidf_cols].values)

    machine_kw_tfidf_cols = [c for c in df.columns if c.startswith('feed_machine_kw_tfidf_')]
    df_machine_kw = df[~df['feed_machine_kw_tfidf_0'].isnull()]
    run_kmeans(df_machine_kw[machine_kw_tfidf_cols].values)


    # 1.2 author聚类
    author_emb = df.drop_duplicates(subset='authorid')

    author_desc_tfidf_cols = [c for c in author_emb.columns if c.startswith('author_description_tfidf_')]
    author_desc_emb = author_emb[~author_emb['author_description_tfidf_0'].isnull()][author_desc_tfidf_cols]
    run_kmeans(author_desc_emb.values)

    author_manu_tag_tfidf_cols = [c for c in author_emb.columns if c.startswith('author_manu_tag_tfidf_')]
    author_manu_tag_emb = author_emb[~author_emb['author_manu_tag_tfidf_0'].isnull()][author_manu_tag_tfidf_cols]
    run_kmeans(author_manu_tag_emb.values)

    author_machine_tag_tfidf_cols = [c for c in author_emb.columns if c.startswith('author_machine_tag_tfidf_')]
    author_machine_tag_emb = author_emb[~author_emb['author_machine_tag_tfidf_0'].isnull()][author_machine_tag_tfidf_cols]
    run_kmeans(author_machine_tag_emb.values)


    author_manu_kw_tfidf_cols = [c for c in author_emb.columns if c.startswith('author_manu_kw_tfidf_')]
    author_manu_kw_emb = author_emb[~author_emb['author_manu_kw_tfidf_0'].isnull()][author_manu_kw_tfidf_cols]
    run_kmeans(author_manu_kw_emb.values)


    author_machine_kw_tfidf_cols = [c for c in author_emb.columns if c.startswith('author_machine_kw_tfidf_')]
    author_machine_kw_emb = author_emb[~author_emb['author_machine_kw_tfidf_0'].isnull()][author_machine_kw_tfidf_cols]
    run_kmeans(author_machine_kw_emb.values)


#####################################################################################
##################### 2、feed、author 聚类 ##############################################
# 利用前面生成的 tfidf 特征对 feed、author 进行聚类

def clustering(arr, k):
    """对数据进行聚类，获取每个数据的所属类别
    """
    cluster = KMeans(n_clusters=k, random_state=123)
    pred = cluster.fit_predict(arr)
    return pred

def get_cluster(df, col_prefix, res_col, k, cluster_col='feedid'):
    df_new = df[~df[col_prefix+'0'].isnull()]
    print(col_prefix+'0', df_new.shape)
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    cls_res = clustering(df_new[cols].values, k=k)
    new_col = res_col+'_'+str(k)
    df_new[new_col] = cls_res
    print(df_new.shape)
    return df_new[[cluster_col, new_col]]

# feed 各个特征的聚类数量
feed_clusters = [
    ('feed_manu_tag_tfidf_',32),
    ('feed_machine_tag_tfidf_',32),
    ('feed_manu_kw_tfidf_',22),
    ('feed_machine_kw_tfidf_',17),
    ('feed_description_tfidf_', 18)
]

# author 各个特征的聚类数量
author_clusters = [
    ('author_manu_tag_tfidf_',19),
    ('author_machine_tag_tfidf_',21),
    ('author_manu_kw_tfidf_',18),
    ('author_machine_kw_tfidf_',18),
    ('author_description_tfidf_',18)
]


def get_all_cluster(df, cls_dict, cluster_col='feedid'):
    df_cls_res = df[[cluster_col]]
    for col, k in cls_dict:
        cluster_res = get_cluster(df, col, col+'cls', k, cluster_col)
        df_cls_res = df_cls_res.merge(cluster_res, on=cluster_col, how='left').fillna(k+1)
    
    return df_cls_res


df = pd.read_pickle(f'{DATA_ROOT}/feedid_text_features/feed_author_text_features_fillna_by_author_rm0.pkl')
# feed聚类
feed_cls_res = get_all_cluster(df, feed_clusters, cluster_col='feedid')

# author聚类
author_cls_res = get_all_cluster(df.drop_duplicates(subset=['authorid']), author_clusters, cluster_col='authorid')

df_clusters = df.merge(feed_cls_res, on='feedid', how='left')\
                .merge(author_cls_res, on='authorid', how='left')

cluster_cols = [i[0]+'cls_'+str(i[1]) for i in feed_clusters+author_clusters]
print("CLUSTER COLS:")
print(cluster_cols)
topic_cols = [i for i in df.columns if i.endswith('_topic_class')]
print("TPOIC COLS:")
print(topic_cols)

df_clusters[cluster_cols+topic_cols] = df_clusters[cluster_cols+topic_cols].fillna(-1).astype(int)
reduce_mem_usage(df_clusters).to_pickle(f'{DATA_ROOT}/feedid_text_features/feed_author_text_features_fillna_by_author_clusters.pkl')

# CLUSTER COLS:
# ['feed_manu_tag_tfidf_cls_32', 'feed_machine_tag_tfidf_cls_32', 'feed_manu_kw_tfidf_cls_22', 'feed_machine_kw_tfidf_cls_17', 'feed_description_tfidf_cls_18', 'author_manu_tag_tfidf_cls_19', 'author_machine_tag_tfidf_cls_21', 'author_manu_kw_tfidf_cls_18', 'author_machine_kw_tfidf_cls_18', 'author_description_tfidf_cls_18']
# TPOIC COLS:
# ['feed_manu_tag_topic_class', 'feed_machine_tag_topic_class', 'feed_manu_kw_topic_class', 'feed_machine_kw_topic_class', 'feed_description_topic_class', 'author_description_topic_class', 'author_manu_kw_topic_class', 'author_machine_kw_topic_class', 'author_manu_tag_topic_class', 'author_machine_tag_topic_class']