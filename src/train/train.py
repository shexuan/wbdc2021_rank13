import os, sys, gc, pickle
import preprocess
sys.path.append('../')
from model.moe import MOE
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import random

import logging
logger = logging.getLogger(__name__)


def train_single_model(args, np_rd_seed=2345, rd_seed=2345, torch_seed=1233):
    np.random.seed(np_rd_seed)
    random.seed(rd_seed)
    moe = MOE(dnn_hidden_units=args['hidden_units'], linear_feature_columns=args['linear_feature_columns'],
              dnn_feature_columns=args['dnn_feature_columns'],task='binary',dnn_dropout=args['dropout'],
              l2_reg_embedding=args['l2_reg_embedding'], l2_reg_dnn=args['l2_reg_dnn'],
              device=device, seed=torch_seed, num_tasks=args['num_tasks'],
              pretrained_user_emb_weight=[user_emb_weight],
              pretrained_author_emb_weight=[author_emb_weight],
              pretrained_feed_emb_weight=[feed_emb_weight, official_feed_weight],
              )

    moe.compile(optimizer=args['optimizer'], learning_rate=args['learning_rate'], 
                loss="binary_crossentropy", 
                metrics=["binary_crossentropy",'auc','uauc'])

    metric = moe.fit(online_train_loader, validation_data=None,
                       epochs=args['epochs'], val_userid_list=None,
                       lr_scheduler=args['lr_scheduler'], scheduler_epochs=args['scheduler_epochs'],
                       scheduler_method=args['scheduler_method'], num_warm_epochs=args['num_warm_epochs'],
                       reduction=args['reduction'],
                       task_dict=args['task_dict'], task_weight=args['task_weight'],verbose=2,
                       early_stop_uauc=0.55)
    torch.save(moe.state_dict(), f'{MODEL_SAVE_PATH}/npseed{np_rd_seed}_rdseed{rd_seed}_torchseed{torch_seed}')
    del moe
    gc.collect()
    torch.cuda.empty_cache()


if __name__=='__main__':
    global MODEL_SAVE_PATH
    DATA_PATH = '/home/tione/notebook/wbdc2021-semi/data/my_data/'
    MODEL_SAVE_PATH = '/home/tione/notebook/wbdc2021-semi/data/model/'
    pretrained_models = {
        'sg_ns_64_epoch30':{
            'official_feed': f'{DATA_PATH}/official_feed_emb.d512.pkl',
            'feedid': f'{DATA_PATH}/feedid_w7_iter10.64d.filled_cold.pkl',
            'authorid': f'{DATA_PATH}/authorid_w7_iter10.64d.filled_cold.pkl',
            'userid_by_feed': f'{DATA_PATH}/userid_by_feedid_w10_iter10.64d.pkl',
        }
    }

    USED_FEATURES = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id','videoplayseconds_bin','bgm_na',
                     'videoplayseconds','tag_manu_machine_corr']+\
                    ['feed_machine_tag_tfidf_cls_32','feed_machine_kw_tfidf_cls_17',
                     'author_machine_tag_tfidf_cls_21','author_machine_kw_tfidf_cls_18']

    args = {}
    args['USED_FEATURES'] = USED_FEATURES
    args['DATA_PATH'] = DATA_PATH

    global hidden_units
    hidden_units = (784,512,128)
    args['hidden_units'] = hidden_units
    args['batch_size'] = 40000
    args['emb_dim'] = 128
    args['learning_rate'] = 0.06
    args['lr_scheduler'] = True
    args['epochs'] = 2
    args['scheduler_epochs'] = 3
    args['num_warm_epochs'] = 0
    args['scheduler_method'] = 'cos'
    args['use_bn'] = True
    args['reduction'] = 'sum'
    args['optimizer'] = 'adagrad'
    args['num_tasks'] = 7
    args['early_stop_uauc'] = 0.689
    args['num_workers'] = 7
    args['dropout'] =  0.0
    args['l2_reg_dnn'] = 0.001
    args['l2_reg_embedding'] = 0.01
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

    args['pretrained_model'] = pretrained_models['sg_ns_64_epoch30']
    
        
    logger.info("Parameters: ")
    logger.info(args)
    
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
    args['linear_feature_columns'] = linear_feature_columns
    args['dnn_feature_columns'] = dnn_feature_columns
    lbe_dict = preprocess.LBE_MODEL

    pri_train_X = pickle.load(open(DATA_PATH+'/pri_train_x.pkl','rb'))
    pri_train_y = pickle.load(open(DATA_PATH+'/pri_train_y.pkl','rb'))
    pri_val_X = pickle.load(open(DATA_PATH+'/pri_val_x.pkl','rb'))
    pri_val_y = pickle.load(open(DATA_PATH+'/pri_val_y.pkl','rb'))

    semi_train_X = pickle.load(open(DATA_PATH+'/semi_train_x.pkl','rb'))
    semi_train_y = pickle.load(open(DATA_PATH+'/semi_train_y.pkl','rb'))
    semi_val_X = pickle.load(open(DATA_PATH+'/semi_val_x.pkl','rb'))
    semi_val_y = pickle.load(open(DATA_PATH+'/semi_val_y.pkl','rb'))

    # 从数据集中选取部分特征
    semi_train_X = {f.name:semi_train_X[f.name] for f in dnn_feature_columns}
    semi_val_X = {f.name:semi_val_X[f.name] for f in dnn_feature_columns}
    pri_train_X = {f.name:pri_train_X[f.name] for f in dnn_feature_columns}
    pri_val_X = {f.name:pri_val_X[f.name] for f in dnn_feature_columns}
    
    # 复赛+初赛数据
    for col in semi_train_X:
        semi_train_X[col] = np.concatenate((semi_train_X[col], pri_train_X[col]), axis=0)
    semi_train_y = np.concatenate((semi_train_y, pri_train_y), axis=0)
    
    # 载入label encoder模型
    LBE_MODEL_PATH = f'{DATA_PATH}/lbe_dic_all.pkl'
    lbe_dict = pickle.load(open(LBE_MODEL_PATH, 'rb'))
    
    global user_emb_weight, author_emb_weight, feed_emb_weight, official_feed_weight
    # 载入预训练Embedding weight matrix
    user_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['userid'], 
                                                        args['pretrained_model']['userid_by_feed'], padding=True)
    # user_by_author_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['userid'], 
    #                                                     args['pretrained_model']['userid_by_author'], padding=True)

    author_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['authorid'], 
                                                        args['pretrained_model']['authorid'], padding=True)
    feed_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
                                                        args['pretrained_model']['feedid'], padding=True)
    # feed_emb_weight_eges = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
    #                                                     '../my_data/eges/feedid_eges0_emb.pkl', padding=True)
    official_feed_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
                                                        args['pretrained_model']['official_feed'], padding=True)

    logger.info('All used features:')
    logger.info(semi_train_X.keys())

    device = 'gpu'
    if device=='gpu' and torch.cuda.is_available():
        # print('cuda ready...')
        device = 'cuda:1'
    else:
        device = 'cpu'

    _moe = MOE(dnn_hidden_units=args['hidden_units'], linear_feature_columns=linear_feature_columns,
              dnn_feature_columns=dnn_feature_columns, task='binary', dnn_dropout=0.,
              l2_reg_embedding=0., l2_reg_dnn=0.,
              l2_reg_linear=0., device=device, seed=1233, num_tasks=args['num_tasks'],
              pretrained_user_emb_weight=None,
              pretrained_author_emb_weight=None,
              pretrained_feed_emb_weight=None,
              )

    # 用于线上预测的训练集, 初赛+复赛+初复赛验证集
    online_train_X = {}
    for col in semi_train_X:
        online_train_X[col] = np.concatenate((semi_train_X[col], semi_val_X[col], pri_val_X[col]), axis=0)
    online_train_y = np.concatenate((semi_train_y, semi_val_y, pri_val_y), axis=0)

    online_train_loader = preprocess.get_dataloader(online_train_X, _moe, y=online_train_y, 
                                                    batch_size=args['batch_size'],  
                                                    num_workers=7)
    del _moe
    gc.collect()
    
    # 测试
    # train_single_model(args, np_rd_seed=2345, rd_seed=2345, torch_seed=1233)
    for _ in range(50):
        seed1 = random.randint(1, 100000)
        seed2 = random.randint(1, 100000)
        seed3 = random.randint(1, 100000)
        logger.info("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("np random seed = " +str(seed1))
        logger.info("random seed = " +str(seed2))
        logger.info("torch random seed = " +str(seed3))
        train_single_model(args, np_rd_seed=seed1, rd_seed=seed2, torch_seed=seed3)
    
    
    