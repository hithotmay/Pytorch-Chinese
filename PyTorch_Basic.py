import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
# Table of Contents 
# 目录#
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 

# 1.基本自动降级示例 1（第 25 行到 39 行）
# 2.基本自动等级示例 2（第 46 行到 83 行）
# 3.从 numpy 加载数据（第 90 行到 97 号线）
# 4.输入点线（第 104 至 129 行）
# 5.自定义数据集的输入点线（第 136 行到 156 行）
# 6.预训型号（第 163 至 176 行）
# 7.保存和加载型号（线路 183 到 189）


# ================================================================== #
#                     1. Basic autograd example 1                    #
#                     1. 基本自动级示例 1                             #
# ================================================================== #

# Create tensors.
# 创建张量。
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
# 生成计算图。
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
# 计算梯度。
y.backward()

# Print out the gradients.
# 打印梯度。
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 


# ================================================================== #
#                    2. Basic autograd example 2                     #
#                    2.基本自动级示例 2                               #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
# 创建张量 （10， 3） 和 （10， 2）。
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer.
# 构建完全连接层。
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build loss function and optimizer.
# 生成损失函数和优化器。
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
# 前向传递。
pred = linear(x)

# Compute loss.
# 计算损失值。
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass.
# 反向传递。
loss.backward()

# Print out the gradients.
# 打印梯度渐变。
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
# 第一步 梯度下降。
optimizer.step()

# You can also perform gradient descent at the low level.
# 您也可以在低水平执行梯度下降。
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
# 打印第一步梯度下降后的损失。
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
#                     3. 从 numpy 加载数据                            #
# ================================================================== #

# Create a numpy array.
# 创建numpy数组。
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
# 将numpy数组转换为torch张量。
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
# 将torch张量转换为numpy数组。
z = y.numpy()


# ================================================================== #
#                         4. Input pipeline                          #
#                         4. 输入管道                                 #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
# 下载并构造 CIFAR-10 数据集。
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one data pair (read data from disk).
# 获取一个数据对（从磁盘读取数据）。
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
# 数据加载程序（以非常简单的方式提供队列和线程）。
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
# 迭代开始时，队列和线程开始从文件加载数据
data_iter = iter(train_loader)

# Mini-batch images and labels.
# 小批图像和标签。
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
# 数据加载器的实际使用情况如下。
for images, labels in train_loader:
    # Training code should be written here.
    # 训练代码应写在这里。
    pass


# ================================================================== #
#                5. Input pipeline for custom dataset                #
#                5.自定义数据集的输入管道                              #
# ================================================================== #

# You should build your custom dataset as below.
# 应构建如下自定义数据集。
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        # 1.初始化文件路径或文件名列表。
        pass
    
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 1.从文件中读取一个数据（例如使用 numpy.fromfile，PIL.Image.open）。
        # 2.预处理数据（例如torchvision.Transform）。
        # 3.返回数据对（例如图像和标签）。
        pass
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        # 应将 0 更改为数据集的总大小。
        return 0 

# You can then use the prebuilt data loader. 
# 然后可以使用预构建的数据加载程序。
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
#                        6. 预训练模型                                #
# ================================================================== #

# Download and load the pretrained ResNet-18.
# 下载并加载预先训练的 Resnet-18。
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
# 如果只想微调模型的顶层，请设置如下。
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
# 更换顶层进行微调。
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
# 前向传递。
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
#                      7. 保存并加载模型                              #
# ================================================================== #

# Save and load the entire model.
# 保存并加载整个模型。
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
# 仅保存并加载模型参数（建议）。
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
