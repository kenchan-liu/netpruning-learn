import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

device = torch.device('cuda')
# 载入训练集
train_dataset = datasets.MNIST(root='./MNIST/',
                               train=True, # 载入训练集
                               transform=transforms.ToTensor(), # 转变为tensor数据
                               download=True)       # 下载数据
#载入测试集
test_dataset = datasets.MNIST(root='./MNIST/',
                               train=False, # 载入测试集
                               transform=transforms.ToTensor(), # 转变为tensor数据
                               download=True)       # 下载数据

# 设置批次大小（每次传入数据量）
batch_size = 64 # 每次训练64张图片的数据

# 装载数据集
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, #每批数据的大小
                          shuffle=True) # shuffle表示打乱数据
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size, #每批数据的大小
                          shuffle=True) # shuffle表示打乱数据

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        weight_params1 = torch.nn.init.xavier_uniform_(torch.Tensor(16,1,5,5))
        bias_params1 = torch.zeros((16,),requires_grad=True)
        self.conv1_weight = nn.Parameter(weight_params1)
        self.conv1_bias = nn.Parameter(bias_params1)
        weight_params2 = torch.nn.init.xavier_uniform_(torch.Tensor(32,16,5,5))
        bias_params2 = torch.zeros((32,),requires_grad=True)
        self.conv2_weight = nn.Parameter(weight_params2)
        self.conv2_bias = nn.Parameter(bias_params2)
        self.fc_weight = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(10,32*7*7)))
        self.fc_bias = nn.Parameter(torch.randn((10,)),requires_grad=True)

        self.conv1_weight=self.conv1_weight.to(device)
        self.conv2_weight=self.conv2_weight.to(device)
        self.conv1_bias=self.conv1_bias.to(device)
        self.conv2_bias=self.conv2_bias.to(device)
        self.fc_weight=self.fc_weight.to(device)
        self.fc_bias=self.fc_bias.to(device)

        self.sparsity=0.5
        #mask矩阵，用于剪枝
        self.register_buffer('conv1_mask', torch.ones((16,1,5,5),dtype=torch.uint8))
        self.register_buffer('conv2_mask', torch.ones((32,16,5,5),dtype=torch.uint8))

    def forward(self,x):
        #更新卷积层1的mask矩阵
        w1=self.conv1_weight.clone().detach()
        w1=torch.where(self.conv1_mask==1,w1,torch.zeros(w1.size()))
        w1=torch.abs(w1)
        sorted,indices=torch.sort(w1.view(-1),descending=False)
        threshold1=sorted[int(sorted.size(0)*self.sparsity)]
        self.conv1_mask=torch.tensor(w1.ge(threshold1),dtype=torch.uint8)
        #print(torch.sum(self.conv1_mask))
        #更新卷积层2的mask矩阵
        w2 = self.conv2_weight.clone().detach()
        w2 = torch.where(self.conv2_mask == 1, w2, torch.zeros(w2.size()))
        w2 = torch.abs(w2)
        sorted, indices = torch.sort(w2.view(-1), descending=False)
        threshold2 = sorted[int(sorted.size(0) * self.sparsity)]
        self.conv2_mask = torch.tensor(w2.ge(threshold2), dtype=torch.uint8)
        #第一个卷积层
        self.conv1_weight.data=self.conv1_weight*self.conv1_mask
        x=F.conv2d(input=x,weight=self.conv1_weight,bias=self.conv1_bias,stride=1,padding=2) #1,28,28 ---> 16,28,28
        x=F.relu(x)
        #池化层
        x=F.max_pool2d(x,kernel_size=2,stride=2)                                             #(16,14,14)
        #第二个卷积层
        self.conv2_weight.data=self.conv2_weight*self.conv2_mask
        x=F.conv2d(input=x,weight=self.conv2_weight,bias=self.conv2_bias,stride=1,padding=2) #16,14,14 ---> 32,14,14
        x=F.relu(x)
        #池化层
        x=F.max_pool2d(x,kernel_size=2,stride=2)                        #32,14,14 --》32,7,7
        #
        x=x.view(x.size(0),-1)                                          #展开成(batch_size,32*7*7)
        #全连接层
        x=F.linear(x,self.fc_weight,bias=self.fc_bias)
        x = F.softmax(x, dim=1)
        return x


model = Net()
model.to(device)
#定义代价函数
mse_loss = nn.MSELoss()
#定义优化器
LR=0.01 #学习率
optimizer = optim.SGD(model.parameters(),lr=LR)

def train_model():
    for i, data in enumerate(train_loader):
        # 循环一次获得一批次的数据与标签
        inputs, labels = data
        inputs, labels = inputs.to(device) , labels.to(device)
        # 获得模型预测结果
        out = model(inputs)
        # to onehot,把数据标签变为独热编码
        labels = labels.reshape(-1, 1)  # 将一维数据变为二维数据（64）->(64,1)
        one_hot = torch.zeros(inputs.shape[0], 10,device=device).scatter(1, labels, 1)
        loss = mse_loss(out, one_hot)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 修改权值
        optimizer.step()

def test_model():
    correct = 0
    for i, data in enumerate(test_loader):
        # 获取一批次的数据
        inputs, labels = data
        # 预测结果
        out = model(inputs)
        # 获得最大值即最大值所在的位置
        _, predicted = torch.max(out, 1)
        # 对比预测结果与标签（累积预测正确的数量）
        correct += (predicted == labels).sum()
    print("Test acc:{0}".format(correct.item() / len(test_dataset)))

for epoch in range(20):
    print('epoch:', epoch)
    train_model()
    test_model()

print(model.conv1_weight*model.conv1_mask)
print(model.conv2_weight*model.conv2_mask)
print(torch.sum(model.conv1_mask))
print(torch.sum(model.conv2_mask))
