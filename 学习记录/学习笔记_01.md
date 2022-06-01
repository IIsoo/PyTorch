# Pytorch学习（一）

## 1.安装Python+Pytorch（windows平台)

### （1）安装IDE+Python：<br/>
下载安装[VScode IDE](https://code.visualstudio.com/)，并在仓库搜索Python安装(超级方便！其他IDE自行探索)<br/><br/>


### （2）安装anaconda并配置虚拟环境 ：<br/>
下载安装[anaconda](https://www.anaconda.com/) 并添加环境变量到PATH中（此处环境变量为自己安装anaconda的文件路径）<br/>
![1654045506(1)](https://user-images.githubusercontent.com/31993576/171307981-723ac8a9-783d-4013-9bc4-5549e88cbc42.png)<br/>
打开Anaconda prompt 并在命令行中输入命令`conda create -n PyTorch python=3.x(此处为你安装的python版本)`<br/>回车后就创建了一个名为PyTorch（可自命名）的虚拟环境<br/><br/>

### （4）激活进入虚拟环境：<br/>
接着输入`conda activate PyTorch`进入虚拟环境<br/>
![image](https://user-images.githubusercontent.com/31993576/171304340-8cabd638-a59c-4cdb-ab2b-cd31ef280b17.png)<br/>
当前方括号中变成你的环境名字便是激活成功<br/><br/>

### （5）安装Pytorch：<br/>
**CPU版本-针对电脑没有GPU的学习者：** 打开[Pytorch官网](https://pytorch.org/get-started/locally/)，选择CPU版本并复制下方命令<br/>`conda install pytorch torchvision torchaudio cpuonly -c pytorch`
![image](https://user-images.githubusercontent.com/31993576/171305275-83f954a5-736a-4248-8d41-710e61d258ba.png)<br/><br/>
**GPU版本-针对有独立显卡的电脑：** 首先确定自己安装了显卡驱动，在NVIDIA控制面板中功能菜单中帮助->系统信息->组件中找到自己的GPU版本<br/>
![image](https://user-images.githubusercontent.com/31993576/171308509-3784c6ff-c492-406c-9dd3-79936d6f5a5d.png)
![image](https://user-images.githubusercontent.com/31993576/171308523-97e78ff4-59ff-49ce-8385-8ffe8461eac9.png)<br/>
确定自己的cuda版本后在官网选择对应版本安装代码<br/>
![image](https://user-images.githubusercontent.com/31993576/171308698-096dd8f9-52c4-41a9-8516-e4c79f9b25f3.png)<br/>

然后回车，因为是国外源，下载速度可能很慢或者会断连，多尝试几次<br/><br/>

### （6）测试是否安装成功：<br/>
在anaconda中激活Pytorch环境后输入命令`Python`并回车<br/>
在Python编辑中输入`import torch`回车，如果没有出现报错即安装成功<br/>
![image](https://user-images.githubusercontent.com/31993576/171308302-d19ebbd5-a40e-4261-bbb7-59d98dce74f2.png)

