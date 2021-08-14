# **2021中国高校计算机大赛-微信大数据挑战赛Baseline**

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 

本次比赛以多个行为预测结果的加权uAUC值进行评分。大赛官方网站：https://algo.weixin.qq.com/

## **1. 环境依赖**
- pandas==1.0.5
- numpy==1.19.5
- numba==0.53.1
- scipy==1.5.0
- torch==1.4.0
- python==3.6.5
- gensim==3.8.0
- deepctr-torch==0.2.7
- transformers==3.1.0
- bayesian-optimization==1.2.0
- tensorflow==2.5.0
- tensorflow-estimator==2.5.0

## **2. 目录结构**


```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──Step1_feed_text_process.py
|       ├──Step2_feed_text_cluster.py
|       ├──Step3_w2v_feed_author_user.py
|       ├──Step4_train_eges.py
|       ├──Step5_bayes_smooth.py
│   ├── model, codes for model architecture
|       ├──moe.py  
|   ├── train, codes for training
|       ├──preprocess.py
|       ├──generate_train_data.py 
|       ├──opt_moe.py
|       ├──train.py
|   ├── inference.py, main function for inference on test dataset
|   ├── utils.py, some utils functions
├── data
│   ├── wedata, dataset of the competition
│       ├── wechat_algo_data1, preliminary dataset
│       ├── wechat_algo_data2, preliminary dataset
│   ├── my_data, train data and features for training models
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (e.g. pytorch trained model state dict)

```

## **3. 运行流程**
- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 安装环境：source init.sh
- 预测并生成结果文件：sh inference.sh /home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/test_b.csv
- 数据准备和模型训练：sh train.sh

## **4. 模型及特征**
- 模型：Multi-perceptron DNN
- 参数：
    - batch_size: 40000
    - emded_dim: 128
    - num_epochs: 2
    - learning_rate: 0.06
    
- 特征：userid, feedid, authorid, bgm_singer_id, bgm_song_id, videoplayseconds, feed和author的tag、keyword聚类特征，
      以及user、feed、author的word2vec Embedding特征;


## **5. 算法性能**
- 资源配置：2*P40_48G显存_14核CPU_112G内存
- 预测耗时
    - 单模总预测时长: 3418 s
    - 单个目标行为2000条样本的平均预测时长: 228 ms


## **6. 代码说明**
模型预测部分代码位置如下：

| 路径 | 行数 | 内容 |
| :--- | :--- | :--- |
| src/inference.py | 69 | `pred_arr = moe.predict(test_loader)`|

## **7. 相关文献**
无
