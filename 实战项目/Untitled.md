## 02.基于csv数据集的机器学习

### Ⅰ.导入库与csv数据集


```python
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing

csv_value = pd.read_csv('temps.csv')
```

我们这次除了导入最基本的torch和numpy外，还有处理数据集的pandas和sklearn


```python
csv_value.head(), csv_value.shape
```




    (   year  month  day  week  temp_2  temp_1  average  actual  friend
     0  2016      1    1   Fri      45      45     45.6      45      29
     1  2016      1    2   Sat      44      45     45.7      44      61
     2  2016      1    3   Sun      45      44     45.8      41      56
     3  2016      1    4   Mon      44      41     45.9      40      53
     4  2016      1    5  Tues      41      40     46.0      44      41,
     (348, 9))



输出数据集的头部，可以看到数据集和形状，但是其中有星期的特征，计算机是不能处理字符的，所以我们要先处理一下

### Ⅱ.把数据集预处理成Pytorch可用的格式


```python
csv_value = pd.get_dummies(csv_value) #这个方法能将数据中字符类型的特征转换成数值类型
csv_value.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>temp_2</th>
      <th>temp_1</th>
      <th>average</th>
      <th>actual</th>
      <th>friend</th>
      <th>week_Fri</th>
      <th>week_Mon</th>
      <th>week_Sat</th>
      <th>week_Sun</th>
      <th>week_Thurs</th>
      <th>week_Tues</th>
      <th>week_Wed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>45</td>
      <td>45</td>
      <td>45.6</td>
      <td>45</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>1</td>
      <td>2</td>
      <td>44</td>
      <td>45</td>
      <td>45.7</td>
      <td>44</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>1</td>
      <td>3</td>
      <td>45</td>
      <td>44</td>
      <td>45.8</td>
      <td>41</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>1</td>
      <td>4</td>
      <td>44</td>
      <td>41</td>
      <td>45.9</td>
      <td>40</td>
      <td>53</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016</td>
      <td>1</td>
      <td>5</td>
      <td>41</td>
      <td>40</td>
      <td>46.0</td>
      <td>44</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




接着我们来从中分割出训练集和标签集


```python
valid_value = np.array(csv_value['actual']) #将数据集中的真实温度作为标签集
valid_value = torch.tensor(valid_value, dtype=torch.float32, requires_grad=True).reshape(-1, 1) #转换成tensor格式的列向量并可导
train_value = np.array(csv_value.drop('actual', axis=1)) #把去掉真实温度的数据集作为训练集
train_value = preprocessing.StandardScaler().fit_transform(train_value) #对训练集做标准化预处理
train_value = torch.tensor(train_value, dtype=torch.float32, requires_grad=True) #转化训练集为tensor格式并可导
train_value.data.numpy()[1], valid_value.data.numpy()[1]
```




    (array([ 0.        , -1.5678393 , -1.5426712 , -1.5692981 , -1.4944355 ,
            -1.3375576 ,  0.06187741, -0.40961596, -0.40482044,  2.4413111 ,
            -0.40482044, -0.40482044, -0.41913682, -0.40482044], dtype=float32),
     array([44.], dtype=float32))



训练集和标签集就都被我们处理好了


### Ⅲ.定义我们的训练模型


```python
my_nn = torch.nn.Sequential(
    torch.nn.Linear(train_value.shape[1], 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)
```

这次我们使用了一个两层的模型，包含两个全连接和一个非线性处理ReLu()
输入值是训练集的特征数量，使用128个神经元，输出1个预测值，也就是当天温度

### Ⅳ.定义超参数、损失函数和优化器


```python
epochs = 1000 # 周期1000
batch_size = 16 # batch包16为一个
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001) # 优化器Adam能够动态调节学习率lr
cost = torch.nn.MSELoss() # 损失函数还是用MESloss
```

这里有个新的超参数batch_size，主要作用是将训练集分批次投入模型中训练，使模型参数更新地更频繁，这样的训练效率会更高

### Ⅴ.开始训练


```python
for epoch in range(epochs):
    losses =[] # 增加一个存储loss的数组，用于计算每个周期的损失均值
    epoch += 1
    for start in range(0, len(train_value), batch_size): # 把训练集分成一个个batch进行训练
        end = start + batch_size if start + batch_size < len(train_value) else len(train_value)
        outputs = my_nn(train_value[start: end])
        loss = cost(valid_value[start: end], outputs)
        losses.append(loss.data.numpy()) # 把当前batch的损失值加入数组
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 100 == 0:# 输出损失
        print('{}/{}, Loss={:.2f}'.format(epoch, epochs, np.mean(losses)))
```

    100/1000, Loss=9.49
    200/1000, Loss=9.39
    300/1000, Loss=9.25
    400/1000, Loss=9.27
    500/1000, Loss=8.80
    600/1000, Loss=8.72
    700/1000, Loss=8.10
    800/1000, Loss=7.75
    900/1000, Loss=7.70
    1000/1000, Loss=7.21
    

在运行后会发现这次耗时有明显提升，这是因为我们的数据集和模型层数都有所增加

这次的练习项目相比上次的更有一些实用性，相信你也获得了小小的成就感，期待我们下次的实战项目！
