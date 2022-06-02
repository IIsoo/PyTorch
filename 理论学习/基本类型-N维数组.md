## 数组介绍

### （1）概念介绍 <br/>
**N维数组**是机器学习和神经网络的主要数据结构，不论是图片，表单，文字都会先被我们转换为该形式再由计算机进行处理<br/><br/>
![image](https://user-images.githubusercontent.com/31993576/171518340-291e07a2-c4d1-48ac-9a99-520d61cc761a.png)
![image](https://user-images.githubusercontent.com/31993576/171518357-908e7611-be1b-4f8e-ae48-85521e152df5.png)<br/><br/>

创建数组需要的参数：<br/>1.形状：例3 * 4矩阵<br/>2.数组元素数据类型：例如float32<br/>3.每个元素的值<br/><br/>

### （2）访问数组元素<br/>
1.访问其中一个元素：Num[1, 2]<br/>
2.访问其中一行：Num[1, :]，访问其中一列Num[:, 1]<br/>
3.访问其中的连续子区域：Num[1:3, 1:]<br/>
4.跳跃访问子区域：Num[::3, ::2] (#表示行中每三行访问一个，列中每2列访问一个)<br/><br/>
![image](https://user-images.githubusercontent.com/31993576/171521251-e3d969b5-2e43-4409-b08b-e706a083491e.png)<br/><br/>

### （3）函数介绍<br/>
1. x = torch.tensor() 创建torch最基本的tensor数据类型，可以直接在括号输入数组元素，例如：x = torch.tensor
2. 
