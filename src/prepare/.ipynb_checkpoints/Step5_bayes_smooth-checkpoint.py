# 贝叶斯迭代代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import gc, os, sys
sys.path.append('../')
from collections import Counter

import pandas as pd
import numpy as np
import pickle

import random
import scipy.special as special

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

RAW_DATA_PATH = '../../wbdc2021/data/wedata/wechat_algo_data2/'

class BayesSmoothing(object):
    """
    refs:
        Click-Through Rate Estimation for Rare Events in Online Advertising
        http://www.cnblogs.com/bentuwuying/p/6498370.html
    """

    def __init__(self, alpha=1, beta=50):
        self.alpha = alpha
        self.beta = beta

    def fit(self, tries, success, iter_num=2000, epsilon=0.00000001):
        """
        fit
        params:
            tries     impression_nums
            success   click_nums
        return:
            alpha, beta
        """
        assert(len(tries) == len(success))
        self._update_from_data_by_moment(tries, success)
        self._update_from_data_by_FPI(tries, success, iter_num, epsilon)
        return self.alpha, self.beta

    def predict(self, tries, success):
        """
        predict
        params:
            tries     impression_nums
            success   click_nums
        return:
            ctrs
        """
        if isinstance(tries, int) and isinstance(success, int):
            return (float(success) + self.alpha) / (float(tries) + self.alpha +
                    self.beta)

        assert(len(tries) == len(success))
        res = []
        for i in range(len(tries)):
            imp = tries[i]
            click = success[i]
            res.append((float(click) + self.alpha) / (float(imp) + self.alpha +
                    self.beta))
        return res

    def _update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''
        estimate alpha, beta using fixed point iteration
        '''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
            logger.info(f'{i}/{iter_num} alpha={new_alpha}, beta={new_beta}')

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''
        fixed point iteration
        '''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def _update_from_data_by_moment(self, tries, success):
        '''
        estimate alpha, beta using moment estimation
        '''
        mean, var = self.__compute_moment(tries, success)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''
        moment estimation
        '''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)
        return mean, var/(len(ctr_list)-1)

def run_smooth(itemid='feedid', act='read_comment', beta=100, save_path=None, val_date_=14):
    feed =  pd.read_csv(f'{RAW_DATA_PATH}/feed_info.csv', header=0)[['feedid', 'authorid']]
    user_act = pd.read_csv(f'{RAW_DATA_PATH}/user_action.csv', header=0)
    user_act = user_act[["userid", "feedid","date_","read_comment",
                         "like","click_avatar","forward","comment","follow","favorite"]]
    
    # 暂不考虑测试集，仅测试在验证集上面的效果
    # test_sub = pd.read_csv('wechat_algo_data1/test_a.csv', header=0)
    # all_act = user_act.append(test_sub).fillna(0).astype(int)
    all_act = user_act.copy()
    
    all_act = all_act.merge(feed, on='feedid')
    all_act['userid_authorid'] = all_act['userid'].astype(str)+'_'+all_act['authorid'].astype(str)
    
    user_act = user_act.query(f'date_<{val_date_}').merge(feed, on='feedid')
    user_act['userid_authorid'] = user_act['userid'].astype(str)+'_'+user_act['authorid'].astype(str)
    
    # 构建item_id转化率特征
    # 训练集+测试集的所有出现过的id
    item_all_list = list(set(all_act[itemid].values))
    # 训练集中的所有出现过的id和其曝光次数
    dic_i = dict(Counter(user_act[itemid].values))
    # 训练集中所有出现过的有点击/行为的id和其行为次数
    dic_cov = dict(Counter(user_act.query(f'{act}==1')[itemid].values))
    # 训练集中所有出现过的id
    l = list(set(user_act[itemid].values))
    
    del user_act, feed, all_act
    gc.collect()
    
    I = []
    C = []
    for item in l:
        I.append(dic_i[item]) #该item被浏览了多少次

    for item in l:
        if item not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[item]) #该item成交次数记录

    
    bs = BayesSmoothing(1, beta)
    bs.fit(I, C)
    # smooth_result = hyper.predict(I, C)
    dic_smooth = {}
    
    for item in item_all_list:
        if item not in dic_i:
            dic_smooth[item] = (bs.alpha) / (bs.alpha + bs.beta)
        elif item not in dic_cov:
            dic_smooth[item] = (bs.alpha) / (dic_i[item] + bs.alpha + bs.beta)
        else:
            dic_smooth[item] = (dic_cov[item] + bs.alpha) / (dic_i[item] + bs.alpha + bs.beta)
    
    
    pickle.dump(dic_smooth, open(os.path.join(save_path, f'{itemid}_{act}_smooth.pkl'), 'wb'))
    
    # dic_smooth = pickle.load(save_path, 'rb')
    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="the function of this module.")
    parser.add_argument("--item", "-i", type=str, help="smooth id")
    parser.add_argument("--action", "-a", type=str, help="smooth act")

    args = vars(parser.parse_args())

    # 各个action的正负样本比例
    args['userid_act_ratio'] = {'read_comment': 29, 'like': 27, 'click_avatar': 96, 
                                'forward': 192, 'favorite': 608, 'comment': 2036, 'follow': 900}
    
    args['feedid_act_ratio'] = {'read_comment': 33, 'like': 48, 'click_avatar': 143, 'forward': 339, 
                                'favorite': 706, 'comment': 2316, 'follow': 1230}
    
    args['authorid_act_ratio'] = {'read_comment': 36, 'like': 60, 'click_avatar': 150, 'forward': 420, 
                                  'favorite': 891, 'comment': 2924, 'follow': 1082}
    
    outpath = 'my_data2/smooth_ctr/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    item = args['item']
    run_smooth(itemid=args['item'], act=args['action'], 
               beta=args[f'{item}_act_ratio'], save_path=outpath)

    # demo 
#     print("bayes_smoothing")
#     hyper = BayesSmoothing(1, 100)
#     I = [73, 2709, 67, 158, 118]
#     C = [0, 30, 2, 3, 4]
#     alpha, beta = hyper.fit(I, C)
#     print("alpha: {0}, beta {1}".format(alpha, beta))
#     print(hyper.predict(I, C))
    
    
