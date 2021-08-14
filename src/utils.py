import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from gensim.models import Word2Vec
import pickle
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
import gensim

@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def auc(actual, predicted):
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)

def uAUC(y_true, y_pred, userids, k=1, backend='loky', weight=[4, 3, 2, 1, 1, 1, 1]):
    num_labels = y_pred.shape[1]

    def uAUC_infunc(i):
        uauc_df = pd.DataFrame()
        uauc_df['userid'] = userids
        uauc_df['y_true'] = y_true[:, i]
        uauc_df['y_pred'] = y_pred[:, i]

        label_nunique = uauc_df.groupby(by='userid')['y_true'].transform('nunique')
        uauc_df = uauc_df[label_nunique == 2]

        aucs = uauc_df.groupby(by='userid').apply(
            lambda x: auc(x['y_true'].values, x['y_pred'].values))
        return np.mean(aucs)

    uauc = Parallel(n_jobs=k, backend='loky')(delayed(uAUC_infunc)(i) for i in range(num_labels))
    return np.average(uauc, weights=weight), uauc

# def _uAUC(labels, preds, user_id_list):
#     """Calculate user AUC"""
#     user_pred = defaultdict(lambda: [])
#     user_truth = defaultdict(lambda: [])
#     for idx, truth in enumerate(labels):
#         user_id = user_id_list[idx]
#         pred = preds[idx]
#         truth = labels[idx]
#         user_pred[user_id].append(pred)
#         user_truth[user_id].append(truth)

#     user_flag = defaultdict(lambda: False)
#     for user_id in set(user_id_list):
#         truths = user_truth[user_id]
#         flag = False
#         # 若全是正样本或全是负样本，则flag为False
#         for i in range(len(truths) - 1):
#             if truths[i] != truths[i + 1]:
#                 flag = True
#                 break
#         user_flag[user_id] = flag

#     total_auc = 0.0
#     size = 0.0
#     for user_id in user_flag:
#         if user_flag[user_id]:
#             auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
#             total_auc += auc
#             size += 1.0
#     user_auc = float(total_auc)/size
#     return user_auc


def compute_weighted_score(score_dict, weight_dict):
    '''基于多个行为的uAUC值，计算加权uAUC
    Input:
        scores_dict: 多个行为的uAUC值映射字典, dict
        weights_dict: 多个行为的权重映射字典, dict
    Output:
        score: 加权uAUC值, float
    '''
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict[action])
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    score = round(score, 6)
    return score


def reduce_mem_usage(df, verbose=True):
    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print('~> Memory usage of dataframe is {:.3f} MG'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int' or np.all(np.mod(df[col], 1) == 0):
                # Booleans mapped to integers
                if list(df[col].unique()) == [1, 0]:
                    # df[col] = df[col].astype(bool)
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # df[col] = df[col].astype('category')
            pass

    if verbose:
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print(
            '~> Memory usage after optimization is: {:.3f} MG'.format(end_mem))
        print('~> Decreased by {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))
        print('---' * 20)
    return df


def apply_by_group(grouped, f):
    """
    Applies a function to each DataFrame in a DataFrameGroupBy object, concatenates the results
    and returns the resulting DataFrame.

    Parameters
    ----------
    grouped: DataFrameGroupBy
        The grouped DataFrame that contains column(s) to be ranked and, potentially, a column with weights.
    f: callable
        Function to apply to each DataFrame.

    Returns
    -------
    DataFrame that results from applying the function to each DataFrame in the DataFrameGroupBy object and
    concatenating the results.

    """
    assert isinstance(grouped, pd.core.groupby.DataFrameGroupBy)
    assert hasattr(f, '__call__')

    data_frames = []
    for key, data_frame in grouped:
        data_frames.append(f(data_frame))
    return pd.concat(data_frames)

# apply_by_group(tmp_user_act2[['userid', cate]+cols].groupby(['userid', cate]), lambda x: x.fillna(x.mean()))


def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).
    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with gensim.utils.open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))


def dict2model(df, idcol, cols, save_name='feedid_raw_manu_tag_tfidf'):
    # df = df.query(f'{cols[0]}=={cols[0]}').drop_duplicates(subset=[idcol])
    emb_dic = dict(zip(list(df[idcol].astype(str)), df[cols].values))
    m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=len(cols))
    m.vocab = emb_dic
    m.vectors = np.array(list(emb_dic.values()))
    my_save_word2vec_format(binary=True, fname='train.bin',
                            total_vec=len(emb_dic), vocab=m.vocab, vectors=m.vectors)
    
    m2 = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('train.bin', binary=True)
    pickle.dump(m, open(save_name, 'wb'))
    
    return pickle.load(open(save_name, 'rb'))