import os, sys, gc, pickle
sys.path.append('../')
from model.moe import MOE
import preprocess

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

import logging
logger = logging.getLogger(__name__)


def process_pipe(feed_path, user_act_path, used_columns, used_sparse_cols, used_dense_cols, 
                 emb_dim=16, is_training=True, test_data=False):
    data = preprocess.preprocess(feed_path, user_act_path)
    data_ds = preprocess.down_sample(data, used_columns, sample_method=None, 
                          neg2pos_ratio=300, user_samp='random', 
                          by_date=None, is_training=is_training)
    if(list(data_ds.head(2)['date_'])[0]==15): # test data
        X_dic, y_arr, linear_feats, dnn_feats, lbe_dict = preprocess.process_features(
                         data_ds, used_sparse_cols, used_dense_cols, 
                         actions=ACTIONS, emb_dim=emb_dim, use_tag_text=None, use_kw_text=None, 
                         feed_history=None, author_history=None,  use_din=False, 
                         max_seq_length=128, behavior_feature_list=['feedid','authorid'],
                         )
        return [(X_dic, y_arr, linear_feats, dnn_feats, lbe_dict)]
    else: # train data
        train_data = data_ds.query('date_<14')
        val_data = data_ds.query('date_==14')
        X_dic_train, y_arr_train, linear_feats, dnn_feats, lbe_dict = preprocess.process_features(
                         train_data, used_sparse_cols, used_dense_cols, 
                         actions=ACTIONS, emb_dim=emb_dim, use_tag_text=None, use_kw_text=None, 
                         feed_history=None, author_history=None,  use_din=False, 
                         max_seq_length=128, behavior_feature_list=['feedid','authorid'],
                         )
        X_dic_val, y_arr_val, linear_feats, dnn_feats, lbe_dict = preprocess.process_features(
                         val_data, used_sparse_cols, used_dense_cols, 
                         actions=ACTIONS, emb_dim=emb_dim, use_tag_text=None, use_kw_text=None, 
                         feed_history=None, author_history=None,  use_din=False, 
                         max_seq_length=128, behavior_feature_list=['feedid','authorid'],
                         )
        return [(X_dic_train, y_arr_train, linear_feats, dnn_feats, lbe_dict),
                (X_dic_val, y_arr_val, linear_feats, dnn_feats, lbe_dict)]


if __name__=='__main__':
    CLS_COLS = ['feed_manu_tag_tfidf_cls_32', 'feed_machine_tag_tfidf_cls_32', 'feed_manu_kw_tfidf_cls_22', 
                'feed_machine_kw_tfidf_cls_17', 'feed_description_tfidf_cls_18', 'author_manu_tag_tfidf_cls_19', 
                'author_machine_tag_tfidf_cls_21', 'author_manu_kw_tfidf_cls_18', 'author_machine_kw_tfidf_cls_18', 
                'author_description_tfidf_cls_18']

    TOPIC_COLS = ['feed_manu_tag_topic_class', 'feed_machine_tag_topic_class', 'feed_manu_kw_topic_class', 
                  'feed_machine_kw_topic_class', 'feed_description_topic_class', 'author_description_topic_class', 
                  'author_manu_kw_topic_class', 'author_machine_kw_topic_class', 'author_manu_tag_topic_class', 
                  'author_machine_tag_topic_class']

    SPARSE_COLS = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id','videoplayseconds_bin','bgm_na']+\
        CLS_COLS+TOPIC_COLS
    DENSE_COLS = ['videoplayseconds','tag_manu_machine_corr']
    ACTIONS = ["read_comment","like","click_avatar","forward",'favorite','comment','follow']

    USED_COLUMNS = SPARSE_COLS + DENSE_COLS + ACTIONS

    DATA_PATH_pri = '/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data1/'
    DATA_PATH_semi = '/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/'
    DATA_PATH = '/home/tione/notebook/src/my_data/'
    OUTPATH = '/home/tione/notebook/wbdc2021-semi/data/my_data/'

    user_act_pri_path = DATA_PATH_pri + '/user_action.csv'
    user_act_semi_path = DATA_PATH_semi + '/user_action.csv'
    test_semi_path = DATA_PATH_semi + '/test_a.csv'
    feed_path = DATA_PATH + '/feedid_text_features/feed_author_text_features_fillna_by_author_clusters.pkl'
    
    semi_train, semi_val = process_pipe(
        feed_path, user_act_semi_path, USED_COLUMNS, SPARSE_COLS, DENSE_COLS)
    pickle.dump(semi_train[0], open(f'{OUTPATH}/semi_train_x.pkl', 'wb'))
    pickle.dump(semi_train[1], open(f'{OUTPATH}/semi_train_y.pkl', 'wb'))

    pickle.dump(semi_val[0], open(f'{OUTPATH}/semi_val_x.pkl', 'wb'))
    pickle.dump(semi_val[1], open(f'{OUTPATH}/semi_val_y.pkl', 'wb'))