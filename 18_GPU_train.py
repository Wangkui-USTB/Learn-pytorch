#   GPU训练：
# 方式1：.cuda()
#          网络模型
#          数据：输入图像，标签
#          损失函数
# 实例1：
# loss_fnu=loss_fnu.cuda()
# images, targets = images.cuda(), targets.cuda()
# loss_fnu=loss_fnu.cuda()

# 方式2：.to(device)
#        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data = torchvision.datasets.CIFAR10("dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
print(f"train_data:{len(train_data)},test_data:{len(test_data)}")
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 搭建网络
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.model = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2),
		                           nn.MaxPool2d(2),
		                           nn.ReLU(),
		                           nn.Conv2d(32,32,5,1,2),
		                           nn.MaxPool2d(2),
		                           nn.ReLU(),
		                           nn.Conv2d(32, 64, 5, 1, 2),
		                           nn.MaxPool2d(2),
		                           nn.ReLU(),
		                           nn.Flatten(),
		                           nn.Linear(1024,64),
		                           nn.ReLU(),
		                           nn.Linear(64,10)
		                           )
	def forward(self,x):
		x=self.model(x)
		return x
# # 测试网络
# model=Model()
# print(model(torch.ones((64,3,32,32))).shape)

# 定义损失函数
loss_fnu=nn.CrossEntropyLoss()
# loss_fnu=loss_fnu.cuda()
loss_fnu=loss_fnu.to(device)
#定义优化器
model=Model()
# model=model.cuda()
model=model.to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

# 设置训练网络参数

epcohs=30
total_train_step=0
total_test_step=0
sum_train_loss=0
for i in range(epcohs):
	# 训练代码
	print(f"--------------第{i+1}轮训练开始--------------")
	sum_train_loss=0
	for data in train_dataloader:
		images,targets=data
		# images, targets=images.cuda(),targets.cuda()to(device)
		images, targets = images.to(device), targets.to(device)
		# print("targets",targets)
		output=model(images)
		loss_value=loss_fnu(output,targets)
		optimizer.zero_grad()
		loss_value.backward()
		optimizer.step()
		sum_train_loss+=loss_value
		total_train_step+=1
		# print(f"第训练{total_train_step}时，损失为{loss_value.item()}")
	print(f"训练第{i+1}轮时，训练损失为{sum_train_loss/len(train_dataloader)}")
	# 测试代码
	sum_test_loss=0
	sum_accuracy=0
	with torch.no_grad():
		for data in test_dataloader:
			images,targets=data
			# images, targets = images.cuda(), targets.cuda()
			images, targets = images.to(device), targets.to(device)
			output=model(images)
			loss_value=loss_fnu(output,targets)
			sum_test_loss+=loss_value
			accuracy=(output.argmax(1)==targets).sum()
			sum_accuracy+=accuracy
	print(f"训练第{i+1}轮后，测试集损失为{sum_test_loss/len(test_dataloader)},分类准确率为：{sum_accuracy/len(test_dataloader)}")
	torch.save(model,"train_save_model/model_CIFAR10_"+str(i+1)+'.pth')
	print("模型保存完毕！")