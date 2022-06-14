## Pytorch代码复用整理

### 1.数据读取预处理

**csv文件处理方式


```python
#读取csv文件
import pandas as pd
data = pd.read_csv(DATA_FILENAME)

#文本类数据过多预处理方法
def data_processing(data, labels_colname): #参数：数据data与标签列名称labels_colname
    data_labels = np.array(data[labels_colname]).reshape(-1, 1)
    data = data.drop('Id', axis=1)# 删除Id列和标签列
    data = data.drop(labels_colname, axis=1)
    num_features = data.dtypes[data.dtypes != 'object'].index #分离出数值类列
    data = data[num_features]
    data_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #将nan值用平均值填充
    data = data_mean.fit_transform(data)
    data = np.array(data)
    data = preprocessing.StandardScaler().fit_transform(data) #对数据进行标准化预处理
    return torch.tensor(data, dtype=torch.float, requires_grad=True).to(device),
            torch.tensor(data_labels, dtype=torch.float, requires_grad=True).to(device) #返回处理好的训练数据和标签
    
#文本类数据较少预处理方法
def data_processing(data, labels_colname): #参数：数据data与标签列名称labels_colname
    data_labels = np.array(data[labels_colname]).reshape(-1, 1)
    data = data.drop('Id', axis=1)# 删除Id列和标签列
    data = data.drop(labels_colname, axis=1)
    data = pd.get_dummies(data) #one-hot方法转化文本数据为数值特征
    data_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #将nan值用平均值填充
    data = data_mean.fit_transform(data)
    data = np.array(data)
    data = preprocessing.StandardScaler().fit_transform(data) #对数据进行标准化预处理
    return torch.tensor(data, dtype=torch.float, requires_grad=True).to(device),
            torch.tensor(data_labels, dtype=torch.float, requires_grad=True).to(device) #返回处理好的训练数据和标签

#将数据打乱并按批量返回
def load_data(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```


### 2.模型训练以及K折验证


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU运算

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(cost(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

#训练模型
def train(net, train_features, train_labels, test_features, test_labels,
          optimizer, cost, epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_data((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
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

#K折验证并输出图像和损失
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

train_l, valid_l = k_fold(k, train_value, train_labels, epochs, lr,
                          wd, batch_size)
print(f'{5}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
```

### 3.保存和读取模型参数


```python
net = net()
torch.save(net.state_dict(), 'mlp.params')
clone.load_state_dict(torch.load('mlp.params'))
```
