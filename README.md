# 实验五：卷积神经网络
<div>
    <span style="float: left">学生姓名: 彭艺琼</span>
    <span style="float: right">指导老师: 胡政伟</span>
</div>
<center>学号: 202410315019 </center>
<br>

## 一、实验目的
掌握基于PyTorch搭建卷积神经网络的方法，复现LeNet‑5模型完成MNIST手写数字识别，理解卷积、池化、激活函数在卷积网络中的作用，观察模型训练收敛过程。

## 二、实验环境
- 编程语言：Python 3
- 深度学习框架：PyTorch、torchvision
- 开发工具：Jupyter Notebook
- 实验数据集：采用MNIST手写数字数据集。

## 三、实验原理
- LeNet‑5是经典卷积神经网络，网络依次由卷积层、激活函数、池化层、全连接层构成。
- 输入为单通道手写数字灰度图像，卷积层利用5×5卷积核提取图像局部纹理与边缘信息，本实验第一层卷积输出6通道特征图，第二层卷积输出16通道特征图。
- 卷积后使用Sigmoid激活引入非线性表达，再通过2×2平均池化压缩特征图尺寸，降低计算开销。
- 经过两次卷积‑激活‑池化之后将多维特征展平，送入全连接层完成特征融合，最后输出10维结果实现0‑9数字分类。训练使用交叉熵损失函数，Adam优化器更新网络权重。

## 四、实验内容
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.clf = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.clf(x)
        return x
```
- 编写代码加载MNIST数据集，设置batch_size=512，搭建LeNet‑5网络结构，配置损失函数与优化器，设置训练轮次为30轮。
- 训练循环内完成前向计算、损失反向传播、参数更新，每轮结束在测试集上计算测试损失与识别准确率，训练结束绘制损失、准确率变化曲线。
- 调试过程中出现优化器获取空参数列表的报错，经排查是Jupyter内核缓存以及代码缩进问题，重启内核并保证网络层定义位于类的构造函数内部后程序正常执行。

## 五、实验结果
- <img width="200" height="401" alt="0d79fd6e206ca93be1453a07f9ecae10" src="https://github.com/user-attachments/assets/786c8c4f-abd0-416b-87bb-27d0aa9e91dd" />
- 模型一共训练30轮，初始epoch=0时测试损失为2.3017，准确率仅0.1135，此时网络权重随机初始化，识别效果接近随机猜测。
- 随着训练迭代，损失快速下降，准确率持续上升，到epoch=10时损失降到0.0714，准确率达到0.9774；
- 训练至第29轮，测试损失稳定在0.0524，测试准确率达到0.9850。
- 从输出日志可以看出，前期模型收敛速度很快，训练后期损失和准确率变化幅度变小，模型逐渐趋于稳定。
- <img width="832" height="286" alt="45350cf55b048c81798c1a72b8c09ad4" src="https://github.com/user-attachments/assets/b9e246e4-564a-45f6-ba82-b2ec18686ac4" />


## 六、实验分析与总结
- 卷积神经网络依靠权值共享的特性，相比纯全连接网络参数量更少，更适合图像任务。
- 本次实验使用Sigmoid激活，最终模型在测试集上取得98.50%的识别准确率，效果较好，但Sigmoid存在梯度饱和的问题，训练后期提升速度变慢，如果更换为ReLU激活，通常可以进一步加快收敛、提升识别效果。
- 通过本次实验完整走完CNN图像分类流程，熟悉数据集加载、网络定义、训练评估整套代码逻辑，对卷积层、池化层、激活函数的实际工作效果建立直观认识。
