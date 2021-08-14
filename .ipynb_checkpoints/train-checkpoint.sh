# 特征处理
cd /home/tione/notebook/wbdc2021-semi/src/prepare/
python Step1_feed_text_process.py && \
python Step2_feed_text_cluster.py && \
python Step3_w2v_feed_author_user.py && \

# 生成训练数据
cd /home/tione/notebook/wbdc2021-semi/src/train/
python generate_train_dat.py && \

# 训练最终模型
python train.py

