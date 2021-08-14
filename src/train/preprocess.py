import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import os, sys, pickle, gc
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
sys.path.append('../')
from model.moe import MOE
import torch.utils.data as Data
from torch.utils.data import DataLoader
from utils import reduce_mem_usage
import logging
logger = logging.getLogger(__name__)

global EMB_DICT
DATA_ROOT = '../my_data/'
RAW_DATA_PATH = '../../wbdc2021/data/wedata/wechat_algo_data2/'
RAW_DATA_PATH_pri = '../../wbdc2021/data/wedata/wechat_algo_data1/'

# LBE_MODEL_PATH = '../my_data/label_encoder_models/lbe_dic_all.pkl'
# if not os.path.exists(os.path.dirname(LBE_MODEL_PATH)):
#     os.makedirs(os.path.dirname(LBE_MODEL_PATH))

# try:
#     LBE_MODEL = pickle.load(open(LBE_MODEL_PATH, 'rb'))
# except:
#     LBE_MODEL = None

EMB_DICT = {
    'w2v_sg_ns_64':{
        'feedid': f'{DATA_ROOT}/w2v_models_sg_ns_64/feedid_w7_iter10.64d.pkl',
        'authorid': f'{DATA_ROOT}/w2v_models_sg_ns_64/authorid_w7_iter10.64d.pkl',
        'userid': f'{DATA_ROOT}/w2v_models_sg_ns_64/userid_w10_iter10.64d.pkl',
        'bgm_singer_id': f'{DATA_ROOT}/w2v_models_sg_ns_64/bgm_singer_id_w7_iter10.64d.pkl',
        'bgm_song_id': f'{DATA_ROOT}/w2v_models_sg_ns_64/bgm_song_id_w7_iter10.64d.pkl',
        'manu_tag': f'{DATA_ROOT}/w2v_models_sg_ns_64/feed_manu_tag_emb_df.64d.pkl',
        'manu_kw': f'{DATA_ROOT}/w2v_models_sg_ns_64/feed_manu_kw_emb_df.64d.pkl',
        'machine_kw': f'{DATA_ROOT}/w2v_models_sg_ns_64/feed_machine_kw_emb_df.64d.pkl',
        'machine_tag': f'{DATA_ROOT}/w2v_models_sg_ns_64/feed_machine_tag_emb_df.64d.pkl'
    }
}

HIST_EMB = {
    3:{
        'feedid':f'{DATA_ROOT}/hist_embed/user_feed_hist3.pkl',
        'authorid':f'{DATA_ROOT}/hist_embed/user_author_hist3.pkl'
    },
    5:{
        'feedid':f'{DATA_ROOT}/hist_embed/user_feed_hist5.pkl',
        'authorid':f'{DATA_ROOT}/hist_embed/user_author_hist5.pkl'
    }
}

TAG_KW_DIM = {
    'manual_tag_list': (534,11), # (unique words, max_length)
    'machine_tag_list': (9382,11),
    'manual_keyword_list': (31131,18),
    'machine_keyword_list': (17960,16)
}

CLS_COLS = ['feed_manu_tag_tfidf_cls_32', 'feed_machine_tag_tfidf_cls_32', 'feed_manu_kw_tfidf_cls_22', 
            'feed_machine_kw_tfidf_cls_17', 'feed_description_tfidf_cls_18', 'author_manu_tag_tfidf_cls_19', 
            'author_machine_tag_tfidf_cls_21', 'author_manu_kw_tfidf_cls_18', 'author_machine_kw_tfidf_cls_18', 
            'author_description_tfidf_cls_18']

TOPIC_COLS = ['feed_manu_tag_topic_class', 'feed_machine_tag_topic_class', 'feed_manu_kw_topic_class', 
              'feed_machine_kw_topic_class', 'feed_description_topic_class', 'author_description_topic_class', 
              'author_manu_kw_topic_class', 'author_machine_kw_topic_class', 'author_manu_tag_topic_class', 
              'author_machine_tag_topic_class']
    
pos_expr = "(read_comment==1)|(like==1)|(click_avatar==1)|(forward==1)|(favorite==1)|(comment==1)|(follow==1)"
neg_expr = "(read_comment!=1)&(like!=1)&(click_avatar!=1)&(forward!=1)&(favorite!=1)&(comment!=1)&(follow!=1)"
ACTIONS = ["read_comment","like","click_avatar","forward",'favorite','comment','follow']

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

def get_emb(fpath, name):
    mm = pickle.load(open(fpath, 'rb'))
    vocabs = list(mm.wv.vocab.keys())
    dic = {}
    for word in vocabs:
        dic[word] = mm.wv.get_vector(word)

    df = pd.DataFrame(dic).T.reset_index()
    df.columns = [name]+[f'{name}_emb_{i}' for i in range(df.shape[1]-1)]
    df[name] = df[name].astype(float).astype(int)
    df = reduce_mem_usage(df)
    
    return df, list(df.columns[1:]), mm

def generate_encoder_models(outfile):
    """生成所有category特征的label encoder模型
    """
    import pickle
    lbe_dic = {}
    # 用户需要考虑复赛的用户和初赛训练集的用户，因为初赛训练集可以用来训练
    user_act = pd.read_csv(f'/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/user_action.csv', 
                           header=0)[['userid', 'device']]\
        .drop_duplicates(subset=['userid','device'])
    user_act_a = pd.read_csv(f'/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data1/user_action.csv', 
                             header=0)[['userid', 'device']]\
        .drop_duplicates(subset=['userid','device'])
    user_act = user_act.append(user_act_a)
    for c in ['userid','device']:
        lbe = LabelEncoder()
        lbe.fit(user_act[c].astype(str))
        lbe_dic[c] = lbe
        print(c, len(lbe.classes_))

    # feed相关特征
    feed_cols = ['feedid','authorid','bgm_song_id','bgm_singer_id','bgm_na','videoplayseconds_bin']
    feed_cols += (CLS_COLS+TOPIC_COLS)
    feed_info = pd.read_pickle(f'{DATA_ROOT}/feedid_text_features/feed_author_text_features_fillna_by_author_clusters.pkl')[feed_cols]

    for c in feed_cols:
        lbe = LabelEncoder()
        lbe.fit(feed_info[c].astype(str))
        lbe_dic[c] = lbe
        print(c, len(lbe.classes_))
    print(outfile)
    pickle.dump(lbe_dic, open(outfile, 'wb'))
    return lbe_dic
    

def preprocess(feed_path, user_act_path, down_sample_neg_by_testid=False, drop_dup_action=True, log_trans=False, 
               discretize=False, fill_bgm_na=False):
    """预处理，合并表
    down_sample_neg_by_testid: 是否根据测试集userid, feedid来进行下采样.
    """
    feed_cols = ['feedid','authorid','bgm_song_id','bgm_singer_id','videoplayseconds_bin','bgm_na',
                 'videoplayseconds','tag_manu_machine_corr']
    feed_cols += (CLS_COLS+TOPIC_COLS)
    # feed_info = pd.read_pickle(f'{DATA_ROOT}/feedid_text_features/feed_author_text_features_fillna_by_author_clusters.pkl')[feed_cols]
    feed_info = pd.read_pickle(feed_path)[feed_cols]
    print("Prepeocessing >>> >>>")
    print("feed number:", feed_info.shape[0])
    if discretize: # 离散化
        feed_info['videoplayseconds_bin'] = feed_info['videoplayseconds'].apply(set_bins)
    # 连续数据进行log转换
    if log_trans:
        feed_info['videoplayseconds'] = np.log(feed_info['videoplayseconds']+1)
        # 对feed_info中的videoplayseconds进行归一化，对'bgm_song_id', 'bgm_singer_id'进行缺失值填充
        mms = MinMaxScaler(feature_range=(0, 1))
        feed_info['videoplayseconds'] = mms.fit_transform(feed_info[['videoplayseconds']])
    
    # 使用authorid来填充bgm缺失
    if fill_bgm_na:
        feed_info.loc[feed_info.bgm_song_id.isna(), 'bgm_song_id'] = \
            (feed_info.loc[feed_info.bgm_song_id.isna(), 'authorid']*-1).astype(int)
        feed_info.loc[feed_info.bgm_singer_id.isna(), 'bgm_singer_id'] = \
            (feed_info.loc[feed_info.bgm_singer_id.isna(), 'authorid']*-1).astype(int)
        feed_info[['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']] = \
           feed_info[['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']].astype(int)
    
    # 特征工程生成的特征的缺失值直接用-1来进行填充
    feed_info[CLS_COLS+TOPIC_COLS] = feed_info[CLS_COLS+TOPIC_COLS].fillna(-1).astype(int)
    reduce_mem = True
    user_act = pd.read_csv(user_act_path, header=0)
    if 'date_' not in user_act.columns:
        user_act['date_'] = 15
        drop_dup_action = False
        reduce_mem = False

    for act_ in ACTIONS:
        if act_ not in user_act.columns:
            user_act[act_] = 0
    
    user_act = user_act[["userid","feedid","date_","device","read_comment",
                         "like","click_avatar","forward",'favorite','comment','follow']]
    print("user actions number: ", user_act.shape[0])
    if reduce_mem:
        user_act = reduce_mem_usage(user_act)
    # 对user_act进行去重操作(另一个可以考虑的点是多次观看为有操作可视为极负样本，增加权重)：
    # （1）多次重复正负行为仅保留一次；
    # （2）若同一user-feed组合同时存在正负行为，则仅保留正行为；
    if drop_dup_action:
        user_act = user_act.sort_values(
            by=['date_','userid','feedid',"read_comment", "like", "click_avatar",
                "forward",'favorite','comment','follow'], ascending=False)

        print("raw user_act number: ", user_act.shape[0])
        user_act_unique = user_act.drop_duplicates(subset=['userid','feedid'], keep='first').drop(columns=ACTIONS)

        print("user_act_unique number: ", user_act_unique.shape[0])
        user_act_sum = user_act.groupby(['userid','feedid'])[ACTIONS].sum().reset_index()
        for act in ["read_comment","like","click_avatar","forward",'favorite','comment','follow']:
            user_act_sum.loc[user_act_sum[act]>=1, act] = 1
        print("user_act_sum number: ", user_act_sum.shape[0])
        user_act = user_act_unique.merge(user_act_sum, on=['userid', 'feedid'], how='inner')
        print("dropped duplicates user_act numbers: ", user_act.shape[0])
    
    df_tot = user_act.merge(feed_info, on='feedid', how='left')
    print("total data size: ", df_tot.shape[0])
    
    print("Preprocessing Done <<< <<< <<<")
    if down_sample_neg_by_testid:  # 训练集仅使用测试集出现过的id
        test_feed = set(df_test['feedid'])
        test_user = set(df_test['userid'])
        trainid_in_test = df_tot.query('date_<15').query(f'(feedid in @test_feed) | (userid in @test_user)')
        return trainid_in_test.append(df_tot.query('date_==15'))
    else:
        return df_tot
        

def down_sample(df, used_columns, sample_method=None, neg2pos_ratio=300, user_samp='random', 
                by_date=None, is_training=True):
    """下采样
    sample_method: 2种采样方式，随机负下采样和根据user进行下采样。根据user进行下采样能保证每个user的样本数量大致相同。
    """
    # 避免更改原used_columns 列表
    used_columns = used_columns[:]+['date_']
    if list(df.head(5)['date_'])[0]==15:  # 测试集直接不做抽样处理
        return df
    if by_date is not None:
        df = df.query(f'date_>={by_date}')
    if sample_method is None: # 
        return df[used_columns]
    elif sample_method=='random':
        print("Sample_method == random")
        df_val = df.query('date_==14').reset_index(drop=True)
        df_pos = df.query(f'({pos_expr}) & (date_<14)')
        df_neg = df.query(f'({neg_expr}) & (date_<14)')

        sample_num = len(df_pos)*neg2pos_ratio
        assert len(df_pos)*neg2pos_ratio <= len(df_neg), \
            f"Negative sample number({len(df_neg)}), is not enough for sampling({sample_num})!"
        df_neg = df_neg.sample(n=sample_num, random_state=234)
        df_train = df_neg.append(df_pos).sample(frac=1., random_state=234).reset_index(drop=True)
        return df_train[used_columns].append(df_val[used_columns])
    else: # sample_method=='user'
        print("Sample_method == user")
        tmp_train = df.query('date_<=14')
        tmp_train['feed_views'] = tmp_train.groupby('feedid')['date_'].transform('count')
        # 训练时第14天不进行下采样，因为他是作为验证集的,测试时第14天进行下采样
        if is_training:
            df_val = tmp_train.query('date_==14').reset_index(drop=True)
            df_pos = tmp_train.query(f'({pos_expr}) & (date_<14)')
            df_neg = tmp_train.query(f'({neg_expr}) & (date_<14)')
            if user_samp=='random':
                df_neg = df_neg.sample(frac=1., random_state=123).groupby('userid').head(neg2pos_ratio)
            else:  
                # 1) 按时间排序，取最靠近的那些样本
                # df_neg = df_neg[::-1].groupby('userid').head(neg2pos_ratio)
                # 2) 取非热门feed
                df_neg = df_neg.groupby('userid').apply(
                    lambda x: x.sort_values(by='feed_views', ascending=True).head(neg2pos_ratio))

            df_train = df_pos.append(df_neg).reset_index(drop=True)
    
        # 线上预测时，验证集也一起进行下采样
        else:
            df_pos = tmp_train.query(f'({pos_expr})')
            df_neg = tmp_train.query(f'({neg_expr})')
            df_neg = df_neg.groupby('userid').apply(
                    lambda x: x.sort_values(by='feed_views', ascending=True).head(neg2pos_ratio))

            df_train = df_neg.append(df_pos)
            df_val = tmp_train.query(f'date_==14')
            
        return df_train[used_columns].append(df_val[used_columns])
            

def get_din_history(df, hist_cols=['feedid', 'authorid'], window=5):
    """din历史序列特征
    """
    behavior_feature_list = hist_cols
    hist_seqs = ['hist_'+c for c in behavior_feature_list]
    df = df.sort_values(by='date_', ascending=True)
    df_din = pd.DataFrame()
    for dt in range(df['date_'].min()+1, 15):
        # 需要逆序，如此在后面截断的时候保留的是最近的行为
        pre = df.query(f'(date_<{dt}) & (date_>={dt-window})')[::-1].groupby('userid')[behavior_feature_list]\
                .agg(list).reset_index()
        pre.columns = ['userid'] + hist_seqs 
        pre['seq_length'] = pre[pre.columns[1]].apply(len)
        pre['date_'] = dt
        df_din = df_din.append(pre)
    
    return df_din[['userid','date_','seq_length']+hist_seqs]


def process_features(df, used_sparse_feats, used_dense_feats, actions=ACTIONS, 
                     emb_dim=16, use_tag_text=None, use_kw_text=None, 
                     feed_history=None, author_history=None,  use_din=False, 
                     max_seq_length=128, behavior_feature_list=['feedid','authorid'],
                     ):
    """对特征进行编码，处理出模型所需要的输入
    """
    # 避免更改原输入列表
    used_sparse_feats = used_sparse_feats[:]
    used_dense_feats = used_dense_feats[:]
    used_varlen_feats = []
    
    # 使用历史观看过的id的average Embedding
    if feed_history in [3, 5]:
        feed_hist = pd.read_pickle(HIST_EMB[feed_history]['feedid'])
        df = df.merge(feed_hist, on=['userid','date_'], how='left')
        used_dense_feats.extend([i for i in feed_hist.columns if i.startswith('hist_')])
    if author_history in [3,5]:
        author_hist = pd.read_pickle(HIST_EMB[author_history]['authorid'])
        df = df.merge(author_hist, on=['userid','date_'], how='left')
        used_dense_feats.extend([i for i in author_hist.columns if i.startswith('hist_')])
    
    # 加上 varlength tag、keyword数据
    if use_tag_text:
        feed_text = pd.read_pickle(f'{DATA_ROOT}/feedid_text_features/feed_tag_kw_padded_text.pkl')[['feedid',use_tag_text]]
        df = df.merge(feed_text, on='feedid', how='left')
        used_varlen_feats.append(VarLenSparseFeat(SparseFeat(use_tag_text, TAG_KW_DIM[use_tag_text][0], 
                                                             embedding_dim=emb_dim), 
                                     TAG_KW_DIM[use_tag_text][1]))
    
    if use_kw_text:
        feed_text = pd.read_pickle(f'{DATA_ROOT}/feedid_text_features/feed_tag_kw_padded_text.pkl')[['feedid',use_kw_text]]
        df = df.merge(feed_text, on='feedid', how='left')
        used_varlen_feats.append(VarLenSparseFeat(SparseFeat(use_kw_text, TAG_KW_DIM[use_kw_text][0], 
                                                             embedding_dim=emb_dim), 
                                     TAG_KW_DIM[use_kw_text][1]))

    # 1. Label Encoding for sparse features
    for feat in used_sparse_feats:
        lbe = LBE_MODEL[feat]
        # index 0 作为padding 备用，因此label encoder结果+1
        df[feat] = lbe.transform(df[feat].astype(str))+1

    df = reduce_mem_usage(df)
    
    # 2. count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, len(LBE_MODEL[feat].classes_)+1, emb_dim) for feat in used_sparse_feats] + \
                             [DenseFeat(feat, 1, ) for feat in used_dense_feats]
    
    dnn_feature_columns = fixlen_feature_columns[:] + used_varlen_feats
    linear_feature_columns = fixlen_feature_columns[:]  + used_varlen_feats
    
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    # 3. generate input data for model
    X = {name: df[name].values for name in feature_names}

    for f in used_varlen_feats:
        X[f.name] = np.vstack(df[f.name].values).astype(np.int16)

    return X, df[actions].values, linear_feature_columns, dnn_feature_columns, LBE_MODEL


def load_feature_pretrained_embedding(lbe, model_path, padding=True):
    print(model_path)
    model = pickle.load(open(model_path, 'rb'))
    emb_dim = model.vector_size
    print('classes numbers: ', len(lbe.classes_))
    print('word2vec vocab size: ', len(model.wv.vocab))
    # 因为在process_features函数中label encoder后全部+1了，所以这里的index也要+1
    word2idx = zip(lbe.classes_, lbe.transform(lbe.classes_)+1)
    # 先随机初始化
    weights = np.random.normal(0, 0.001, (len(lbe.classes_)+padding, emb_dim))
    
    random_count = 0
    for word, idx in word2idx:
        try: 
            weights[idx] = model.wv.get_vector(str(word))
        except AttributeError: # dataframe 转换过来的gensim model只能这种形式
            try:
                weights[idx] = model.vocab[str(word)]
            except:
                random_count+=1
        except:
            random_count+=1
            
    print('Total Random initialized word embedding counts: ', random_count)
    
    return weights


def get_dataloader(x, model, y=None, batch_size=4096, num_workers=2, 
                   pretrained_ids=['userid','authorid','feedid']):
    # 生成data_loader
    if isinstance(x, dict):
        x = [x[feature] for feature in model.feature_index]
        
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)
    
    shuffle = True
    if y is not None: # 有y，训练集
        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
    else:
        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        shuffle = False
    # train_tensor_data = BalanceDataset(np.concatenate(x, axis=-1), y)

    train_loader = DataLoader(
        dataset=tensor_data,
        #sampler=ImbalancedDatasetSampler(train_tensor_data),
        shuffle=shuffle, batch_size=batch_size, drop_last=False, 
        pin_memory=True, num_workers=num_workers)
    
    return train_loader


def train_model(model, train_x, train_y, val_x, val_y, test_x=None, 
                batch_size=4096, epochs=3, learning_rate=0.01, optimizer='adagrad',
                lr_scheduler=False, scheduler_epochs=3, scheduler_method='cos',
                reduction='sum', task_dict=None, task_weight=None,num_workers=2,
                pretrained_user_emb_weight=None, pretrained_author_emb_weight=None,
                pretrained_feed_emb_weight=None, pretrained_feed_tag_emb_weight=None,
                pretrained_feed_keyword_emb_weight=None, official_feed_emb_weight=None):
    """训练模型
    """
    val_userid_lst = val_x['userid'].tolist()
    train_loader = get_dataloader(train_x, model, y=train_y, batch_size=batch_size)
    val_x_loader = get_dataloader(val_x, model, batch_size=batch_size)
    
    model.compile(optimizer=optimizer, learning_rate=learning_rate, loss="binary_crossentropy", 
                  metrics=["binary_crossentropy",'auc','uauc'])

    metric = model.fit(train_loader, validation_data=[val_x_loader, val_y],
                       epochs=epochs, val_userid_list=val_userid_lst, 
                       lr_scheduler=lr_scheduler, scheduler_epochs=scheduler_epochs,
                       scheduler_method=scheduler_method, reduction=reduction,
                       task_dict=task_dict, task_weight=task_weight,num_workers=num_workers)
    
    torch.cuda.empty_cache()
    return metric


def train_predict(model, train_x, train_y, val_x, val_y, test_x=None, linear_feats=None, dnn_feats=None, 
                  batch_size=512, epochs=10, learning_rate=0.001, lr_scheduler=True, scheduler_epochs=5,
                  reduction='sum', optimizer='momentum', merge_val=True,
                  pretrained_user_emb_weight=None, pretrained_author_emb_weight=None,
                  pretrained_feed_emb_weight=None, pretrained_feed_tag_emb_weight=None,
                  pretrained_feed_keyword_emb_weight=None, official_feed_emb_weight=None):
    """预测结果提交
    """
    if merge_val:
        X_tot = {}
        for col in train_x:
            X_tot[col] = np.concatenate((train_x[col], val_x[col]), axis=0)
        y_tot = np.concatenate((train_y, val_y), axis=0)
    else:
        X_tot = train_x
        y_tot = train_y
    
    model.compile(optimizer=optimizer, learning_rate=learning_rate, loss="binary_crossentropy",
                  metrics=["binary_crossentropy",'auc','uauc'])
    
    model.fit(X_tot, y_tot, batch_size=batch_size, validation_data=None,
              epochs=epochs, shuffle=True, lr_scheduler=lr_scheduler, 
              scheduler_epochs=scheduler_epochs, reduction=reduction)
    
    pred_ans = model.predict(test_x, 1024)
    
    return pred_ans


def optimize_moe(train_x, train_y, val_x, val_y, linear_feats=None, dnn_feats=None, 
                batch_size=4096, epochs=3, learning_rate=0.005, device=None,
                lr_scheduler=True, scheduler_epochs=3, scheduler_method='cos',
                num_warm_epochs=0, pbounds=None, use_bn=True,
                optimizer='adagrad', hidden_units=None, reduction='sum',
                num_tasks=4, task_dict=None, task_weight=None, num_workers=2,
                pretrained_user_emb_weight=None, pretrained_author_emb_weight=None,
                pretrained_feed_emb_weight=None, pretrained_bgm_singer_emb_weight=None,
                pretrained_bgm_song_emb_weight=None, early_stop_uauc=0.55,
                bounds_transformer=True, opt_iters=[21, 30]):
    """利用bayesopt来寻找最优参数
    """
    _moe = MOE(dnn_hidden_units=hidden_units, linear_feature_columns=linear_feats,
          dnn_feature_columns=dnn_feats, task='binary', dnn_dropout=0.,
          l2_reg_embedding=0., l2_reg_dnn=0.,
          l2_reg_linear=0., device=device, seed=1233, num_tasks=num_tasks,
          pretrained_user_emb_weight=pretrained_user_emb_weight, 
          pretrained_author_emb_weight=pretrained_author_emb_weight,
          pretrained_feed_emb_weight=pretrained_feed_emb_weight,
          )
    
    val_userid_lst = val_x['userid'].tolist()
    train_loader = get_dataloader(train_x, _moe, y=train_y, 
                                  batch_size=batch_size,num_workers=num_workers)
    val_x_loader = get_dataloader(val_x, _moe, 
                                  batch_size=batch_size,num_workers=num_workers)
    
    def nn_cv(dropout, l2_reg_embedding, l2_reg_dnn):
        """寻参."""
        np.random.seed(2345)
        import random
        random.seed(2345)

        model = MOE(dnn_hidden_units=hidden_units, linear_feature_columns=linear_feats, 
                  dnn_feature_columns=dnn_feats, task='binary', dnn_dropout=dropout,
                  l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, dnn_use_bn=use_bn,
                  l2_reg_linear=0., device=device, seed=1233, num_tasks=num_tasks,
                  pretrained_user_emb_weight=pretrained_user_emb_weight, 
                  pretrained_author_emb_weight=pretrained_author_emb_weight,
                  pretrained_feed_emb_weight=pretrained_feed_emb_weight, 
                  pretrained_bgm_singer_emb_weight=pretrained_bgm_singer_emb_weight,
                  pretrained_bgm_song_emb_weight=pretrained_bgm_song_emb_weight
                  )
        
        model.compile(optimizer=optimizer, learning_rate=learning_rate, loss="binary_crossentropy", 
                      metrics=["binary_crossentropy",'auc','uauc'])

        metric = model.fit(train_loader, validation_data=[val_x_loader, val_y],
                           epochs=epochs, val_userid_list=val_userid_lst,
                           lr_scheduler=lr_scheduler, scheduler_epochs=scheduler_epochs,
                           scheduler_method=scheduler_method, num_warm_epochs=num_warm_epochs,
                           reduction=reduction, num_workers=num_workers,
                           task_dict=task_dict, task_weight=task_weight,verbose=2,
                           early_stop_uauc=early_stop_uauc)
        torch.cuda.empty_cache()

        return metric[1]
    
    bounds_transformer = SequentialDomainReductionTransformer() if bounds_transformer else None
    nn_optimizer = BayesianOptimization(
        f = nn_cv,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum
        # is observed, verbose = 0 is silent
        bounds_transformer=bounds_transformer,
        random_state=2021,
    )
    
    # init_points表示初始点，n_iter代表迭代次数（即采样数）
    nn_optimizer.maximize(init_points=opt_iters[0], n_iter=opt_iters[1])
    print(nn_optimizer.max)

    return nn_optimizer


if __name__=='__main__':
    generate_encoder_models(LBE_MODEL_PATH)
