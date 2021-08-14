import pandas as pd
import numpy as np
import os, sys, gc, sys
import pickle
sys.path.append('../')
from utils import reduce_mem_usage

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


FEED_PATH = '../my_data/feedid_text_features/feed_author_text_features_fillna_by_author_clusters.pkl'
PAIR_PATH = '../my_data/eges/feed_pairs_eges_raw.pkl' # '../my_data/eges/feed_pairs_eges.pkl'

LBE_PATH = '../my_data/eges/feed_lbe_dict.pkl'

USED_COLS = ['feedid','authorid']+['feed_machine_tag_tfidf_cls_32','feed_machine_kw_tfidf_cls_17']


def create_embedding_matrix(sparse_columns, varlen_sparse_columns, embed_dim,
                            init_std=0.0001, padding=True, device='cpu', mode='mean'):
    # sparse_columns => dict{'name':vocab_size}
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    padding_idx = 0 if padding else None
    sparse_embedding_dict = {
        feat: nn.Embedding(sparse_columns[feat], embed_dim, padding_idx=padding_idx)
                             for feat in sparse_columns
    }
    
    if varlen_sparse_columns:
        varlen_sparse_embedding_dict = {
            feat:nn.EmbeddingBag(varlen_sparse_columns[feat], embed_dim, padding_idx=padding_idx,
                                 mode=mode) for feat in varlen_sparse_columns
        }
        sparse_embedding_dict.update(varlen_sparse_embedding_dict)
        
    embedding_dict = nn.ModuleDict(sparse_embedding_dict)
    
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
        # nn.init.kaiming_uniform_(tensor.weight, mode='fan_in', nonlinearity='relu')

    return embedding_dict.to(device)


class EGES(nn.Module):
    def __init__(self, sparse_dict, varlen_sparse_dict=None, target_col='sku_id',
                 n_embed=64, k_side=3, noise_dist=None, device='cpu', padding=True):
        """sparse_dict: dict, {feature_name: vocab_size}
        """
        super().__init__()
        self.n_embed = n_embed
        self.k_side = k_side
        self.device = device
        self.padding = padding
        self.target_col = target_col
        self.features = list(sparse_dict.keys())
        if varlen_sparse_dict:
            self.features = self.features + list(varlen_sparse_dict.keys())
        # 如果padding了的话，则负采样出来的index均需要+1
        self.sample_word_offset = 1 if padding else 0
        # input embedding dict, include item and side info
        self.input_embedding_dict = create_embedding_matrix(
            sparse_dict, varlen_sparse_dict, n_embed,
            init_std=0.0001, padding=padding, device=device, mode='mean')
        self.out_embed = nn.Embedding(sparse_dict[target_col], n_embed,
                                      padding_idx=0 if padding else None)
        self.attn_embed = nn.Embedding(sparse_dict[target_col], k_side+1, 
                                       padding_idx=0 if padding else None)
        
        # Initialize out embedding tables with uniform distribution
        nn.init.normal_(self.out_embed.weight, mean=0, std=0.0001)
        nn.init.normal_(self.attn_embed.weight, mean=0, std=0.0001)

        if noise_dist is None:
            # sampling words uniformly
            self.noise_dist = torch.ones(self.n_vocab)
        else:
            self.noise_dist = noise_dist
        self.noise_dist = self.noise_dist.to(device)

    def forward_input(self, input_dict):
        # return input vector embeddings
        # version 0.1  average all field embeddings as input vector embeddings
        # version 0.2 weighted average all field embeddings as input vector embeddings
        embed_lst = []
        for col in self.features:
            if col in input_dict:
                input_vector = self.input_embedding_dict[col](input_dict[col])
                embed_lst.append(input_vector)

        batch_size = input_vector.shape[0]
        # embeds => [batch_size, k_side+1, n_embed]
        embeds = torch.cat(embed_lst, dim=1).reshape(batch_size, self.k_side+1, self.n_embed)
        
        # attation => [batch_size, k_side+1]
        attn_w = self.attn_embed(input_dict[self.target_col])
        attn_w = torch.exp(attn_w)
        attn_s = torch.sum(attn_w, dim=1).reshape(-1, 1)
        attn_w = (attn_w/attn_s).reshape(batch_size, 1, self.k_side+1) # 归一化
        
        # attw => [batch_size, 1, k_side+1]
        # embeds => [batch_size, k_side+1, embed_size]
        # matmul out => [batch_size, 1, embed_size]
        input_vector = torch.matmul(attn_w, embeds).squeeze(1)
        
        return input_vector

    def forward_output(self, output_words):
        # return output vector embeddings 
        output_vector = self.out_embed(output_words)
        return output_vector
    
    def forward_noise(self, batch_size, n_samples):
        """Generate noise vectors with shape [batch_size, n_samples, n_embed]
        """
        # sample words from our noise distribution 
        noise_words = torch.multinomial(self.noise_dist, batch_size*n_samples, 
                                        replacement=True) + self.sample_word_offset
        noise_vector = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vector


class NegativeSamplingLoss(nn.Module):
    """这里用的是负对数似然, 而不是sampled softmax
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape
        
        # input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # target words log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        
        # negative sampling words log-sigmoid loss
        # negative words sigmoid optmize to small, thus here noise_vectors.neg()
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        # sum the losses over the sample of noise vectors
        noise_loss = noise_loss.squeeze().sum(1)
        
        # sum target and negative loss
        return -(out_loss + noise_loss).mean()


class TextData(Dataset):
    def __init__(self, df, sparse_columns=['feedid','label','authorid','feed_machine_tag_tfidf_cls_32',
                                           'feed_machine_kw_tfidf_cls_17'],
                 varlen_sparse_columns=[], device='cpu'):
        self.sparse_columns = sparse_columns
        self.varlen_sparse_columns = varlen_sparse_columns
        self.device = device
        self.data = {
            col:df[col].values for col in sparse_columns
        }
        if varlen_sparse_columns:
            for col in varlen_sparse_columns:
                self.data[col] = np.vstack(df[col].values)

        self.data_num = len(df)
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        data_dic = {}
        for col in self.sparse_columns:
            data_dic[col] = torch.tensor(self.data[col][idx]).long() #.to(self.device)
        if self.varlen_sparse_columns:
            for col in self.varlen_sparse_columns:
                data_dic[col] = torch.tensor(self.data[col][idx, :]).long() #.to(self.device)

        return data_dic
    

def preprocess():
    lbe_dict = pickle.load(open(LBE_PATH, 'rb'))
    df_pair = pd.read_pickle(PAIR_PATH)
    
    # 各个field的维度，包含padding index
    vocab_dict = {feat:len(lbe_dict[feat].classes_)+1 for feat in lbe_dict}
    counts = pickle.load(open('../my_data/eges/feed_raw_counts.pkl', 'rb'))
    noise_dist = torch.from_numpy(counts**(0.75)/np.sum(counts**(0.75)))
    
    return df_pair, vocab_dict, noise_dist


def train():
    df_pair, vocab_dict, noise_dist = preprocess()
    device = 'gpu'
    if device=='gpu' and torch.cuda.is_available():
        # print('cuda ready...')
        device = 'cuda:0'
    else:
        device = 'cpu'

    textdata = TextData(df_pair, sparse_columns=['feedid','label','authorid','feed_machine_tag_tfidf_cls_32',
                                               'feed_machine_kw_tfidf_cls_17']) 
    textloader = DataLoader(textdata,
                            batch_size=10000,
                            shuffle=True,
                            num_workers=10,
                            drop_last=False,
                            pin_memory=True)
    
    embedding_dim = 64
    model = EGES(vocab_dict, n_embed=embedding_dim, k_side=3, target_col='feedid',
                 noise_dist=noise_dist, device=device, padding=True).to(device)
    model = torch.nn.DataParallel(model)
    criterion = NegativeSamplingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    
    logger.info("hello !!!!!!!!!!")
    for e in range(3):
        for i, data_dic in enumerate(textloader):
            # input, output and noise vectors
            data_dic = {feat:data_dic[feat].to(device) for feat in data_dic}
            input_vectors = model.module.forward_input(data_dic)
            output_vectors = model.module.forward_output(data_dic['label'])
            noise_vectors = model.module.forward_noise(data_dic['label'].shape[0], 10)
            # negative sampling loss
            loss = criterion(input_vectors, output_vectors, noise_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%5000==0:
                logger.info(f'Epoch {e}/5 Step {i} Loss = {loss}')
    
        torch.save(model.module.state_dict(), f'../my_data/eges/feed_raw_model_epoch{e}.bin')
    

if __name__=='__main__':
    logger.info("fuck !!!!!!!!!!")
    train()