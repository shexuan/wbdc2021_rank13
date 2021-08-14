import os, sys, gc, pickle
sys.path.append('../')
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


if __name__=='__main__':
    TOPIC_COLS = ['feed_manu_tag_topic_class', 'feed_machine_tag_topic_class', 'feed_manu_kw_topic_class', 
                  'feed_machine_kw_topic_class', 'feed_description_topic_class', 'author_description_topic_class', 
                  'author_manu_kw_topic_class', 'author_machine_kw_topic_class', 'author_manu_tag_topic_class', 
                  'author_machine_tag_topic_class']
    W2V_DIR_EPOCH10 = '../my_data/w2v_models_sg_ns_64_epoch10/'
    W2V_DIR_EPOCH20 = '../my_data/w2v_models_sg_ns_64_epoch20/'
    W2V_DIR_EPOCH30 = '../my_data/w2v_models_sg_ns_64_epoch30/'

    pretrained_models = {
        'sg_ns_64_epoch20':{
            'official_feed': f'../my_data/official_feed_emb.d512.pkl',
            'official_feed_pca': f'../my_data/official_feed_emb_pca.d32.pkl',
            'feedid': f'{W2V_DIR_EPOCH20}/feedid_w7_iter10.64d.pkl',
            'authorid': f'{W2V_DIR_EPOCH20}/authorid_w7_iter10.64d.pkl',
            'userid_by_feed': f'{W2V_DIR_EPOCH20}/userid_by_feedid_w10_iter10.64d.pkl',
            'userid_by_author': f'{W2V_DIR_EPOCH20}/userid_by_authorid_w10_iter10.64d.pkl',
        },
        'sg_ns_64_epoch30':{
            'official_feed': f'../my_data/official_feed_emb.d512.pkl',
            'official_feed_pca': f'../my_data/official_feed_emb_pca.d32.pkl',
            'feedid': f'{W2V_DIR_EPOCH30}/feedid_w7_iter10.64d.pkl',
            'authorid': f'{W2V_DIR_EPOCH30}/authorid_w7_iter10.64d.pkl',
            'userid_by_feed': f'{W2V_DIR_EPOCH30}/userid_by_feedid_w10_iter10.64d.pkl',
            'userid_by_author': f'{W2V_DIR_EPOCH30}/userid_by_authorid_w10_iter10.64d.pkl',
        }
    }

    USED_FEATURES = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id','videoplayseconds_bin','bgm_na',
                     'videoplayseconds','tag_manu_machine_corr']+\
                    ['feed_machine_tag_tfidf_cls_32','feed_machine_kw_tfidf_cls_17',
                     'author_machine_tag_tfidf_cls_21','author_machine_kw_tfidf_cls_18']

    DATA_PATH = '../my_data/data_base/'
    
    args = {}
    args['USED_FEATURES'] = USED_FEATURES
    args['DATA_PATH'] = DATA_PATH

    global hidden_units
    hidden_units = (1024,512,128) # (512,256,128)
    args['hidden_units'] = hidden_units
    args['batch_size'] = 40000
    args['emb_dim'] = 16
    args['learning_rate'] = 0.075
    args['lr_scheduler'] = True
    args['epochs'] = 2
    args['scheduler_epochs'] = 3
    args['num_warm_epochs'] = 0
    args['scheduler_method'] = 'cos'
    args['use_bn'] = True
    args['reduction'] = 'sum'
    args['optimizer'] = 'adagrad'
    args['num_tasks'] = 4
    args['early_stop_uauc'] = 0.69
    args['num_workers'] = 7
    args['task_dict'] = {
            0: 'read_comment',
            1: 'like',
            2: 'click_avatar',
            3: 'forward',
            4: 'favorite',
            5: 'comment',
            6: 'follow'
    }
    args['task_weight'] = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1
    }
    
    args['opt_iters'] = [21, 20]
    args['pbounds'] = {'dropout': (0.0, 0.5),
                       #'learning_rate': 0.001,
                       'l2_reg_dnn': (1e-4, 0.01),
                       'l2_reg_embedding': (1e-4, 0.01),
                      }

    args['bounds_transformer'] = False
    args['pretrained_model'] = pretrained_models['sg_ns_64_epoch30']
    # 全部特征
    linear_feature_columns = pickle.load(open(DATA_PATH+'/linear_feature.pkl','rb'))
    dnn_feature_columns = pickle.load(open(DATA_PATH+'/dnn_feature.pkl','rb'))
    #print('raw:')
    #print(dnn_feature_columns)
    # 使用其中部分特征
    linear_feature_columns = [f for f in linear_feature_columns if f.name in USED_FEATURES]
    dnn_feature_columns = [f for f in dnn_feature_columns if f.name in USED_FEATURES]
    features = []
    for f in linear_feature_columns:
        if isinstance(f, SparseFeat):
            features.append(SparseFeat(f.name, f.vocabulary_size, args['emb_dim']))
        else:
            features.append(f)
    linear_feature_columns = features
    dnn_feature_columns = features
    
    lbe_dict = preprocess.LBE_MODEL

    # pri_train_X = pickle.load(open(DATA_PATH+'/pri_train_x.pkl','rb'))
    # pri_train_y = pickle.load(open(DATA_PATH+'/pri_train_y.pkl','rb'))
    # pri_val_X = pickle.load(open(DATA_PATH+'/pri_val_x.pkl','rb'))
    # pri_val_y = pickle.load(open(DATA_PATH+'/pri_val_y.pkl','rb'))

    semi_train_X = pickle.load(open(DATA_PATH+'/semi_train_x.pkl','rb'))
    semi_train_y = pickle.load(open(DATA_PATH+'/semi_train_y.pkl','rb'))
    semi_val_X = pickle.load(open(DATA_PATH+'/semi_val_x.pkl','rb'))
    semi_val_y = pickle.load(open(DATA_PATH+'/semi_val_y.pkl','rb'))
    # 从数据集中选取部分特征
    semi_train_X = {f.name:semi_train_X[f.name] for f in dnn_feature_columns}
    semi_val_X = {f.name:semi_val_X[f.name] for f in dnn_feature_columns}
    # pri_train_X = {f.name:pri_train_X[f.name] for f in dnn_feature_columns}
    
#     for col in semi_train_X:
#         semi_train_X[col] = np.concatenate((semi_train_X[col], pri_train_X[col]), axis=0)
#     semi_train_y = np.concatenate((semi_train_y, pri_train_y), axis=0)
    

    lbe_dict = preprocess.LBE_MODEL
    # 载入预训练Embedding weight matrix
    user_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['userid'], 
                                                        args['pretrained_model']['userid_by_feed'], padding=True)
    author_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['authorid'], 
                                                        args['pretrained_model']['authorid'], padding=True)
    feed_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
                                                        args['pretrained_model']['feedid'], padding=True)
#     feed_machine_tag_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
#                                                         args['pretrained_model']['feed_machine_tag'], padding=True)
#     feed_manu_tag_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
#                                                         args['pretrained_model']['feed_manu_tag'], padding=True)
#     feed_machine_kw_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
#                                                         args['pretrained_model']['feed_machine_kw'], padding=True)
#     feed_manu_kw_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
#                                                         args['pretrained_model']['feed_manu_kw'], padding=True)
    official_feed_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
                                                        args['pretrained_model']['official_feed'], padding=True)
    
    logger.info('All used features:')
    logger.info(semi_train_X.keys())
    
    device = 'gpu'
    if device=='gpu' and torch.cuda.is_available():
        # print('cuda ready...')
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    opt_res = preprocess.optimize_moe(semi_train_X, semi_train_y,
                  semi_val_X, semi_val_y,
                  linear_feats=linear_feature_columns,
                  dnn_feats=dnn_feature_columns, 
                  batch_size=args['batch_size'], epochs=args['epochs'], 
                  learning_rate=args['learning_rate'], device=device,
                  use_bn=args['use_bn'],
                  lr_scheduler=args['lr_scheduler'], 
                  scheduler_epochs=args['scheduler_epochs'], 
                  scheduler_method=args['scheduler_method'],
                  num_warm_epochs=args['num_warm_epochs'],
                  pbounds=args['pbounds'],
                  optimizer=args['optimizer'],
                  hidden_units=args['hidden_units'],
                  reduction=args['reduction'],
                  num_tasks=args['num_tasks'], 
                  task_dict=args['task_dict'], 
                  task_weight=args['task_weight'],
                  num_workers=args['num_workers'],
                  pretrained_user_emb_weight=[user_emb_weight], 
                  pretrained_author_emb_weight=[author_emb_weight],
                  pretrained_feed_emb_weight=[feed_emb_weight, official_feed_weight],
                  pretrained_bgm_singer_emb_weight=None,
                  pretrained_bgm_song_emb_weight=None,
                  opt_iters = args['opt_iters'],
                  early_stop_uauc=args['early_stop_uauc'],
                  bounds_transformer=args['bounds_transformer'])

    args.update(opt_res.max)
    logger.info('Training Args and Results:')
    logger.info(args)