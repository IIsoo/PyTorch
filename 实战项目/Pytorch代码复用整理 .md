## Pytorch代码复用整理

## Ⅰ.CSV数据代码

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

## Ⅱ.图像数据

### 1.图像数据导入以及预处理


```python
import torch
import pandas as pd
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
import shutil, os
from torch.utils.data import DataLoader, random_split
import random
import time
import copy

data_dir = './data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#创建label文件夹将对应图片存储进去

train_data = pd.read_csv('train.csv')
names = set(train_data['label'])
#创建对应的train和valid文件夹
if os.path.exists('./data') == False:
    for name in names:
        path_train = './data/train/'+str(name)
        path_valid = './data/valid/'+str(name)
        os.makedirs(path_train)
        os.makedirs(path_valid)
    #将图像数据按8：2随机放入数据集中
    for i in range(len(train_data)):
        img_class = train_data.label[i]
        img_file = train_data.image[i]
        img_name = str(img_file).replace('images/', '')
        if(random.randint(1,100)>20):
            shutil.copyfile(img_file, './data/train/'+str(img_class)+'/'+img_name)
        else:
            shutil.copyfile(img_file, './data/valid/'+str(img_class)+'/'+img_name)
        if i%1000==0:
            print('processed:',i)

#读取训练数据和验证数据
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid']}
```


### 2.图像训练代码


```python
filename='checkpoint.pth'
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,filename=filename):
    model.to(device)
    since = time.time()
    best_acc = 0
    #如果存在训练过的模型，则读取模型继续训练
    if os.path.exists('./'+filename) == True:
        checkpoint = torch.load(filename)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()   # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:#resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            #输出用时和当前训练集、验证集损失还有正确率
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            

            # 每次验证集中最好的模型立即保存
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                  'state_dict': model.state_dict(),
                  'best_acc': best_acc,
                  'optimizer' : optimizer.state_dict(),
                }
                torch.save(state, filename)
                print("----模型文件更新成功----")
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                #scheduler.step()
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

```

### 3.kaggle上传预测结果


```python
import torch
from torchvision import transforms, datasets
import torchvision
import os
import pandas as pd
import shutil

#获得对应分类信息
image_datasets = datasets.ImageFolder('./data/valid')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU运算

#将所有预测图片复制到test文件夹
test_img = pd.read_csv('test.csv')
imgs = test_img['image']
file_path = './data/test'
if os.path.exists(file_path) == False:
    os.makedirs(file_path)
    for i in range(len(test_img)):
        img_file = test_img.image[i]
        img_name = str(test_img.image[i]).replace('images/', '')
        shutil.copyfile(img_file, './data/test/'+img_name)

#对预测图片做变换
test_transforms =transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#读取保存好的模型参数
best_acc = 0
model = torchvision.models.resnet34(pretrained=False)
num_fc_ftr = model.fc.in_features
model.fc = torch.nn.Linear(num_fc_ftr, 176)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
filename='checkpoint.pth'
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()#使用预测模式
model.to(device)

import glob
from PIL import Image

#读取文件夹下的所有图片
test_list = glob.glob('./data/test/*.jpg')
test_df = pd.DataFrame()
for i in range(len(test_list)):
    img = Image.open(test_list[i])
    img = test_transforms(img)
    img = torch.unsqueeze(img, dim=0)
    outputs = model(img.to(device))
    _, preds = torch.max(outputs, 1)
    test_df.at[i, 'image'] = str(test_list[i]).replace("./data/test\\","images/")
    test_df.at[i, 'label'] = str([key for key, value in image_datasets.class_to_idx.items() if value == int(preds)]).strip("['']")
    if i%100 == 0:
        print(i,'/',len(test_list))

test_df[['image']] = test_df[['image']].astype(str)
test_df[['label']] = test_df[['label']].astype(str)
test_df.to_csv('result.csv', index=False, header=True)

print(test_df.head())
```
