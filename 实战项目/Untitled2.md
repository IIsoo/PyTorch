## homewordk-1 加州房价预测

### 1.导入并观察数据


```python
from pickletools import optimize
from matplotlib.pyplot import axis
import sklearn
import torch
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.impute import SimpleImputer
from d2l import torch as d2l
```


读取训练数据并看看前几行


```python
train_value = pd.read_csv('train.csv')
```

由上图我们可以看到数据中包含很多数值和文本数据，其中Sold Price是我们需要预测的数据


### 2.对数据进行预处理

我们首先要到导入的数据进行预处理


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #使用GPU运算

#数据预处理函数
def data_processing(value, labels_col):
    labels = np.array(value[labels_col]).reshape(-1, 1) #首先将数据中的labels列提出来
    value = value.drop('Id', axis=1) #删除Id和labels列
    value = value.drop(labels_col, axis=1)
    num_features = value.dtypes[value.dtypes != 'object'].index #由于文本数据太难处理，直接去掉
    value = value[num_features]
    value_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #对数据中的缺失项使用均值填充
    value = value_mean.fit_transform(value)
    value = np.array(value) #训练数据转成np格式
    value = preprocessing.StandardScaler().fit_transform(value) #对训练数据标准化处理，使训练收敛更快
    return torch.tensor(value, dtype=torch.float, requires_grad=True).to(device), torch.tensor(labels, dtype=torch.float, requires_grad=True).to(device)

train_value, train_labels = data_processing(train_value, 'Sold Price') #获得处理后的训练数据和标签数据
train_value[0:4],train_labels[0:4]
```




    (tensor([[ 8.5758e-02, -2.3570e-02, -2.0581e+00,  0.0000e+00, -7.1282e-03,
              -1.7561e-01, -1.6804e-01,  6.4221e-01, -3.4004e-01,  5.5098e-16,
               1.1201e-16,  9.9643e-01, -3.2646e-01,  9.0058e-02,  1.9815e-01,
               1.0964e+00,  1.2497e-16,  3.2818e-01],
             [-2.1247e-01, -2.3165e-02, -3.1073e-01, -1.0793e-01, -6.0529e-03,
              -6.3549e-02, -5.5395e-02, -1.3660e+00, -1.5927e-01, -2.0578e+00,
              -2.9843e-01, -2.2081e+00, -3.2646e-01, -2.5290e-01, -2.7978e-01,
              -3.0087e-01, -5.1510e-01, -1.4280e+00],
             [ 9.4677e-03, -2.2654e-02,  5.6298e-01, -1.2445e+00, -5.7072e-03,
              -1.7561e-01, -1.6804e-01,  0.0000e+00,  0.0000e+00,  5.5098e-16,
               1.1201e-16, -4.7437e-16,  2.2608e+00, -6.6229e-01, -7.1677e-01,
              -4.3212e-01,  1.2497e-16,  9.2595e-01],
             [-6.6823e-02,  0.0000e+00,  5.6298e-01,  1.0286e+00, -3.9046e-03,
              -1.7561e-01, -1.6804e-01,  1.6463e+00, -4.3043e-01,  1.0439e+00,
              -7.5243e-01,  9.9643e-01, -6.4987e-01,  8.8885e-01,  8.1809e-01,
               2.2031e-01,  7.4299e-01, -1.3471e+00]], device='cuda:0',
            grad_fn=<SliceBackward0>),
     tensor([[3825000.],
             [ 505000.],
             [ 140000.],
             [1775000.]], device='cuda:0', grad_fn=<SliceBackward0>))



这样我们就将训练数据集和标签数据集预处理好了


### 3.定义超参数和模型结构


```python
input_size = train_value.shape[1] #输入特征维度是训练集的列数
output_size = 1 #输出预测值为一个标量
epochs = 100 #100个周期
batch_size = 128 #批量大小
hidden_size = 64 #神经元个数
cost = torch.nn.MSELoss() #损失函数使用MSELoss

#构建模型结构
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, output_size)
)

my_nn = my_nn.to(device) #使用GPU训练
```


### 4.开始训练



```python
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(cost(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = cost(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

#使用k折验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = my_nn
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

train_l, valid_l = k_fold(5, train_value, train_labels, epochs, 0.01,
                          0.001, batch_size)
print(f'{5}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File ~\.conda\envs\Pytorch\lib\site-packages\pandas\core\indexes\base.py:3621, in Index.get_loc(self, key, method, tolerance)
       3620 try:
    -> 3621     return self._engine.get_loc(casted_key)
       3622 except KeyError as err:
    

    File ~\.conda\envs\Pytorch\lib\site-packages\pandas\_libs\index.pyx:136, in pandas._libs.index.IndexEngine.get_loc()
    

    File ~\.conda\envs\Pytorch\lib\site-packages\pandas\_libs\index.pyx:142, in pandas._libs.index.IndexEngine.get_loc()
    

    TypeError: '(slice(0, 9487, None), slice(None, None, None))' is an invalid key

    
    During handling of the above exception, another exception occurred:
    

    InvalidIndexError                         Traceback (most recent call last)

    Input In [32], in <cell line: 61>()
         57         print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}, '
         58               f'验证log rmse{float(valid_ls[-1]):f}')
         59     return train_l_sum / k, valid_l_sum / k
    ---> 61 train_l, valid_l = k_fold(5, train_value, train_labels, epochs, 0.01,
         62                           0.001, batch_size)
         63 print(f'{5}-折验证: 平均训练log rmse: {float(train_l):f}, '
         64       f'平均验证log rmse: {float(valid_l):f}')
    

    Input In [32], in k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size)
         45 train_l_sum, valid_l_sum = 0, 0
         46 for i in range(k):
    ---> 47     data = get_k_fold_data(k, i, X_train, y_train)
         48     net = my_nn
         49     train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
         50                                weight_decay, batch_size)
    

    Input In [32], in get_k_fold_data(k, i, X, y)
         32 for j in range(k):
         33     idx = slice(j * fold_size, (j + 1) * fold_size)
    ---> 34     X_part, y_part = X[idx, :], y[idx]
         35     if j == i:
         36         X_valid, y_valid = X_part, y_part
    

    File ~\.conda\envs\Pytorch\lib\site-packages\pandas\core\frame.py:3505, in DataFrame.__getitem__(self, key)
       3503 if self.columns.nlevels > 1:
       3504     return self._getitem_multilevel(key)
    -> 3505 indexer = self.columns.get_loc(key)
       3506 if is_integer(indexer):
       3507     indexer = [indexer]
    

    File ~\.conda\envs\Pytorch\lib\site-packages\pandas\core\indexes\base.py:3628, in Index.get_loc(self, key, method, tolerance)
       3623         raise KeyError(key) from err
       3624     except TypeError:
       3625         # If we have a listlike key, _check_indexing_error will raise
       3626         #  InvalidIndexError. Otherwise we fall through and re-raise
       3627         #  the TypeError.
    -> 3628         self._check_indexing_error(key)
       3629         raise
       3631 # GH#42269
    

    File ~\.conda\envs\Pytorch\lib\site-packages\pandas\core\indexes\base.py:5637, in Index._check_indexing_error(self, key)
       5633 def _check_indexing_error(self, key):
       5634     if not is_scalar(key):
       5635         # if key is not a scalar, directly raise an error (the code below
       5636         # would convert to numpy arrays and raise later any way) - GH29926
    -> 5637         raise InvalidIndexError(key)
    

    InvalidIndexError: (slice(0, 9487, None), slice(None, None, None))



这样就实现了房价预测模型训练
