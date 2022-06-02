### 1.利用torch.tensor()直接构造tensor


```python
import torch

x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
x
```




    tensor([[1, 2, 3, 4],
            [5, 6, 7, 8]])



可以看到我们直接输入了一个二维数组在tensor中



### 2.一些更简便的构造tensor的方法


```python
a = torch.zeros((2, 3, 4))
b = torch.ones((2, 3, 4))
c = torch.arange(24).reshape(2, 3, 4)
a, b, c
```




    (tensor([[[0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]],
     
             [[0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]]]),
     tensor([[[1., 1., 1., 1.],
              [1., 1., 1., 1.],
              [1., 1., 1., 1.]],
     
             [[1., 1., 1., 1.],
              [1., 1., 1., 1.],
              [1., 1., 1., 1.]]]),
     tensor([[[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]],
     
             [[12, 13, 14, 15],
              [16, 17, 18, 19],
              [20, 21, 22, 23]]]))



以上我们通过zeros、ones和arange构造了分别是全0全1和0-23的（2, 3, 4）的三个tensor

### 3.tensor的大小和格式


```python
x.shape
```




    torch.Size([2, 4])




```python
x.numel()
```




    8



上面的shape函数和numel()方法可以分别访问tensor的大小和元素个数


```python
x.reshape(2, 2, 2)
```




    tensor([[[1, 2],
             [3, 4]],
    
            [[5, 6],
             [7, 8]]])



reshape(num1, num2, ...)方法可以将tensor重构成参数内形式的N维数组


### 4.tensor的计算


```python
a = torch.tensor([2., 3, 4, 5, 6])
b = torch.tensor([7., 8, 9, 10, 11])
a + b, a - b, a * b, a / b, a ** b
```




    (tensor([ 9., 11., 13., 15., 17.]),
     tensor([-5., -5., -5., -5., -5.]),
     tensor([14., 24., 36., 50., 66.]),
     tensor([0.2857, 0.3750, 0.4444, 0.5000, 0.5455]),
     tensor([1.2800e+02, 6.5610e+03, 2.6214e+05, 9.7656e+06, 3.6280e+08]))



由结果可以很清楚的看到tensor之间的计算就是对应元素相计算，但如果两个tensor的格式不同呢，请看下例：


```python
a = torch.tensor([[1, 2, 3]])
b = torch.tensor([[4], [5]])
a.shape, b.shape
```




    (torch.Size([1, 3]), torch.Size([2, 1]))



a和b一个是1 * 3的tensor，一个是2 * 1的tensor，它们之间相加会发生什么


```python
a + b
```




    tensor([[5, 6, 7],
            [6, 7, 8]])



可以看到最终结果是一个3 * 2的tensor，我们发现当格式不同时tensor会利用广播机制延伸出最小可相加的格式

tensor之间还可以进行连结


```python
a = torch.arange(12).reshape(-1 ,4)
b = torch.arange(16).reshape(-1, 4)
c = torch.cat((a, b), dim = 0)
c
```




    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]])



利用cat方法可以将tensor在dim维度连结，当然还需要除dim以外的其他维度都相同

还有求和方法sum


```python
torch.sum(c), torch.sum(c, dim=0)
```




    (tensor(186), tensor([36, 43, 50, 57]))



不设置dim参数会将所有元素求和，设置后会只将dim维度的进行求和


### 5.tensor的访问


```python
c[1], c[0:3], c[:, 0:2]
```




    (tensor([4, 5, 6, 7]),
     tensor([[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]),
     tensor([[ 0,  1],
             [ 4,  5],
             [ 8,  9],
             [ 0,  1],
             [ 4,  5],
             [ 8,  9],
             [12, 13]]))



我们通过tensor[num1:num2, num3:num4, ...]访问tensor中的从dim=0的num1到num2和dim=1的num3到num4的元素


```python
c[0, 0] = 99
c
```




    tensor([[99, 10, 10, 10],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]])




```python
c[0:1] = 10
c
```




    tensor([[10, 10, 10, 10],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]])



我们也可以对tensor元素进行单个改写或者批量改写

### 6.tensor与其他格式之间的转换


```python
import numpy
a = numpy.arange(12)
b = torch.tensor(a)
b
```




    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], dtype=torch.int32)



numpy格式可以转换成tensor


```python
a = torch.tensor([12.5])
a.item(), float(a), int(a)
```




    (12.5, 12.5, 12)



单个元素的tensor可以转换成Python的标量
