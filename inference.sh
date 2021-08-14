# 清空submission目录
rm -r data/submission
mkdir -p data/submission

# 读取测试集路径
test_path=$1
echo "Test path: ${test_path}"
test_size=$((`sed -n '$=' $test_path`-1))
echo "Test size: ${test_size}"

# 开始计时
start=$(date +%s)
# 调用模型预测代码
cd /home/tione/notebook/wbdc2021-semi/src/ 
python inference.py ${test_path}
# 结束计时
end=$(date +%s)

# 计算耗时并输出
## 总预测时长（秒）
take=$(( end - start ))
echo "总预测时长: ${take} s"
## 单个目标行为2000条样本的平均预测时长（毫秒）
avg_take=$(echo "${take} ${test_size}"|awk '{print ($1*2000*1000/(7.0*$2))}')
echo "单个目标行为2000条样本的平均预测时长: ${avg_take} ms"
