# 循环神经网络(Recurrent Neural Networks RNN)

'''
应用范围：自然语言处理NLP（主），计算机视觉
'''
# 参考'https://baike.baidu.com/item/循环神经网络'
'''
  循环神经网络（Recurrent Neural Network, RNN）是一类以序列（sequence）数据为输入，
在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network）。

  对循环神经网络的研究始于二十世纪80-90年代，并在二十一世纪初发展为深度学习（deep learning）算法之一，
其中双向循环神经网络（Bidirectional RNN, Bi-RNN）和长短期记忆网络（Long Short-Term Memory networks，LSTM）
是常见的循环神经网络。

  循环神经网络具有记忆性、参数共享并且图灵完备（Turing completeness），
因此在对序列的非线性特征进行学习时具有一定优势。
循环神经网络在自然语言处理（Natural Language Processing, NLP），
例如语音识别、语言建模、机器翻译等领域有应用，也被用于各类时间序列预报。
引入了卷积神经网络（Convoutional Neural Network,CNN）构筑的循环神经网络可以处理包含序列输入的计算机视觉问题。
'''

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
# ‎设备配置‎
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# 超参数‎
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
# ‎MNIST 数据集‎
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
# ‎数据加载器‎
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Recurrent neural network (many-to-one)
# ‎循环神经网络（多对一）
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        # ‎设置初始隐藏和单元格状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        # ‎向前传播 Lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # ‎解码上次时间步骤的隐藏状态‎
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
# ‎损失和优化
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# ‎训练模型‎
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        # 正向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        # 反向传播并优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# ‎测试模型‎
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
# ‎保存模型检查点‎
torch.save(model.state_dict(), 'model.ckpt')
