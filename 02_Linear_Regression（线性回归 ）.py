# 线性回归
'''
应用范围：预测
'''
# 参考'https://baike.baidu.com/item/线性回归'
'''
线性回归有很多实际用途。分为以下两大类： 
1.如果目标是预测或者映射，线性回归可以用来对观测数据集的和X的值拟合出一个预测模型。当完成这样一个模型以后，对于一个新增的X值，在没有给定与它相配对的y的情况下，
可以用这个拟合过的模型预测出一个y值。
2.给定一个变量y和一些变量X1,...,Xp，这些变量有可能与y相关，线性回归分析可以用来量化y与Xj之间相关性的强度，评估出与y不相关的Xj，并识别出哪些Xj的子集包含了关于y的冗余信息。
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper-parameters
# 超参数
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
# 测试数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
# 线性回归模型
model = nn.Linear(input_size, output_size)

# Loss and optimizer
# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
# 训练模型
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    # 将numpy数组转换为torch张量
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    # 反向传播并优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
# 绘制图形
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
# 保存模型检查点
torch.save(model.state_dict(), 'model.ckpt')
