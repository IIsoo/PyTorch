## 05.kaggle—树叶分类

### 1.训练图片数据处理


```python
import os
import pandas as pd
import random
import shutil

train_data = pd.read_csv('train.csv')
names = set(train_data['label'])

#将图片随机按8：2分成train和valid
print("重构训练集...")
if os.path.exists('./data') == False:
    for name in names:
        path_train = './data/train/'+str(name)
        path_valid = './data/valid/'+str(name)
        os.makedirs(path_train)
        os.makedirs(path_valid)
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
```


### 2.训练分类模型


```python
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
import os
from torch.utils.data import DataLoader
import time
import copy

data_dir = './data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

#数据增广
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
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#读取图片数据，需要文件路径下有分类好的train和valid文件夹
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU运算

#使用预设的resnet模型，并将输出改成176类
my_nn = torchvision.models.resnet34(pretrained=False)
num_fc_ftr = my_nn.fc.in_features
my_nn.fc = torch.nn.Linear(num_fc_ftr, 176)
print('Start training...')

running_loss = 0.0
# 模型保存
filename='checkpoint.pth'

# 优化器设置
optimizer_ft = optim.Adam(my_nn.parameters(), lr=0.001)
#scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)#学习率每7个epoch衰减成原来的1/10
criterion = nn.CrossEntropyLoss()

#训练
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,filename=filename):
    model.to(device)
    since = time.time()
    best_acc = 0
    #读取模型参数
    if os.path.exists(filename) == True:
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
            
            #输出当前epoch的损失和准确率
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            

            # 得到最好那次的模型并立即保存
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
                print()
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
    #return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs 

train_model(my_nn, dataloaders, criterion, optimizer_ft, num_epochs=40)
print('Training Finished.')
```


### 3.分类测试集图片并保存csv用于上传kaggle


```python
import torch
from torchvision import transforms, datasets
import torchvision
import os
import pandas as pd
import shutil

image_datasets = datasets.ImageFolder('./data/valid')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU运算

test_img = pd.read_csv('test.csv')
imgs = test_img['image']
file_path = './data/test'
if os.path.exists(file_path) == False:
    print("构建验证集图片...")
    os.makedirs(file_path)
    for i in range(len(test_img)):
        img_file = test_img.image[i]
        img_name = str(test_img.image[i]).replace('images/', '')
        shutil.copyfile(img_file, './data/test/'+img_name)
        print(i,"/",len(test_img))

test_transforms =transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

best_acc = 0
model = torchvision.models.resnet34(pretrained=False)
num_fc_ftr = model.fc.in_features
model.fc = torch.nn.Linear(num_fc_ftr, 176)

#读取model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
filename='checkpoint.pth'
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

model.eval()
model.to(device)

import glob
from PIL import Image

#读取测试图片并预测
test_list = glob.glob('./data/test/*.jpg')
test_df = pd.DataFrame()
print("----开始预测----")
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
![image](https://user-images.githubusercontent.com/31993576/174486217-0c7ad9c6-53ff-4696-a9c4-9dcb9d5fceb6.png)

本次kaggle作业基本上经历了全套的读取图片、训练模型以及预测分类的过程，在结果出来后感觉非常高兴，但是验证集准确率不高，希望以后有所提升
