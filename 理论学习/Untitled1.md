## 感知机和多层感知机

### 1.感知机

**定义：** 感知机在意义上可以理解为神经网络中的一个神经元

感知机可以解决二分类问题，但是只能线性的进行分类，具有局限性


### 2.多层感知机

将多层感知机堆叠起来，然后加入激活函数进行非线性变化，主要的激活函数有Sigmoid、tanh、ReLU，现在主要使用ReLU激活函数，也就是MAX（0,X)
隐藏层的层数和每层的神经元数量都是超参数，需要人为设置


### 3.多层感知机的从零实现代码


```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Input In [1], in <cell line: 1>()
    ----> 1 import torch
          2 from torch import nn
          3 from d2l import torch as d2l
    

    ModuleNotFoundError: No module named 'torch'


首先我们导入库文件，并将Fashion-MNIST数据集 按批量=256 读取进来



```python
input_size = 784
output_size = 10
hidden_size = 256
epochs = 10

w1 = nn.Parameter(
     torch.randn(input_size, hidden_size, requires_grad=True))
b1 = nn.Parameter(
     torch.zeros(hidden_size, require_grad=True))
w2 = nn.Parameter(
     torch.randn(hidden_size, ouput_size, requires_grad=True))
b2 = nn.Parmeter(
     torch.zeros(output_size, requrie))

Params = [w1, b1, w2, b2]
```

定义超参数，并初始化hidden层和输出层的权值和偏置，这里强调权值w不能全部初始化为0，会导致所有权值梯度相同，无法更新



```python
def ReLU(X):
    a = torch.zeros_like(X)
    return max(0, X)
```

构建ReLU函数也就是max(0, X)



```python
def net(X):
    X = X.reshape(-1, input_size)
    H = ReLU(X @ w1 + b1)
    return (H @ w2 + b2)
```

定义多层感知机网络的计算方式



```python
loss = nn.CrossEntropyLoss()
optimizer = troch.optim.SGD(params, lr=0.1)
d2l.train_ch3(net, train_iter, test_iter, loss, epochs, optimizer)
```

使用经典的交叉熵损失函数和SGD优化器，开始训练！
