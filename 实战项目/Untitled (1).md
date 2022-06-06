## 用Fashion-MNIST数据集训练分类模型

### 1.准备数据集


```python
import torch
from torch import nn
import numpy as np
import torchvision
from IPython import display
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

num_inputs = 784
num_outputs = 10
batch_size = 256
epochs = 10
```

首先我们导入库文件，并定义输入特征数和输出分类数量


```python
trans = [transforms.ToTensor()]
trans = transforms.Compose(trans)
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

导入准备好的训练集和测试集，如果本地没有的话就从网上下载


```python
def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                             num_workers=get_dataloader_workers())

```

将mnist_train分批次随机读取，mnist_test分批次读取

### 2.定义模型、损失函数和优化器等


```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition 
```

定义一个softmax方法计算分类概率

定义模型，由一个全连接组成


```python
my_nn = torch.nn.Sequential(
    nn.Linear(784, 10),
)

cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_nn.parameters(), lr=0.01)
```

损失函数为交叉熵损失，优化器SGD


### 3.开始训练

首先我们构造一个可以返回分类正确率的方法


```python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X.reshape(-1, 784)), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```


```python
开始训练
```


```python
def train(epochs, model, train_data, test_data, cost, optimizer):
    print(f'当前-1: 正确率为{evaluate_accuracy(model, test_data)}') #看还没训练的正确率是多少
    for epoch in range(epochs):
        # metric = Accumulator(3)
        for x, y in train_data:
            model.train()
            loss = cost(model(x.reshape(-1, 784)), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'当前{epoch}: 正确率为{evaluate_accuracy(model, test_data)}')

train(epochs, my_nn, train_iter, test_iter, cost, optimizer)
```

    当前-1: 正确率为0.1252
    当前0: 正确率为0.6817
    当前1: 正确率为0.7234
    当前2: 正确率为0.7443
    当前3: 正确率为0.7605
    当前4: 正确率为0.7689
    当前5: 正确率为0.7759
    当前6: 正确率为0.7826
    当前7: 正确率为0.7884
    当前8: 正确率为0.7918
    当前9: 正确率为0.7933
    

这样我们就能使用Fashion数据集训练一个分类模型
