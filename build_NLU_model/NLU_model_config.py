filter_size = [3, 4, 5]  # 卷积核种类数
filter_num = 64  # 卷积核数量
dropout = 0.5
learning_rate = 0.0001  # 学习率
epochs = 20  # 迭代次数
save_dir = './cache/nlu_best_model.bin'  # 模型保存路径
steps_show = 10  # 每10步查看一次训练集loss和mini batch里的准确率
steps_eval = 100  # 每100步测试一下验证集的准确率
early_stopping = 1000  # 若发现当前验证集的准确率在1000步训练之后不再提高 一直小于best_acc,则提前停止训练
batch_size = 16