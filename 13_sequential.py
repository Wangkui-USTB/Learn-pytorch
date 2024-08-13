from torch import nn
import torch
import torchvision
# 写法1：
class Model1(nn.Module):
	def __init__(self):
		super(Model1,self).__init__()
		self.conv1=nn.Conv2d(3,32,5,padding=2)
		self.maxpool1=nn.MaxPool2d(2)
		self.conv2=nn.Conv2d(32,32,5,padding=2)
		self.maxpool2=nn.MaxPool2d(2)
		self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
		self.maxpool3 = nn.MaxPool2d(2)
		self.flatten=nn.Flatten()
		self.linear1 = nn.Linear(1024,64)
		self.linear2 = nn.Linear(64, 10)
	def forward(self,x):
		x=self.conv1(x)
		x=self.maxpool1(x)
		x=self.conv2(x)
		x=self.maxpool2(x)
		x=self.conv3(x)
		x=self.maxpool3(x)
		x=self.flatten(x)
		x=self.linear1(x)
		x=self.linear2(x)
		return x
# 写法2：
class Model2(nn.Module):
	def __init__(self):
		super(Model2,self).__init__()
		self.model1=nn.Sequential(
			nn.Conv2d(3, 32, 5, padding=2),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 32, 5, padding=2),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 5, padding=2),
			nn.MaxPool2d(2),
			nn.Flatten(),
			nn.Linear(1024, 64),
			nn.Linear(64, 10)
		)
	def forward(self,x):
		x=self.model1(x)
		return x
# 写法1：
model1=Model1()
print("model1:",model1)
input1=torch.ones((64,3,32,32))
print("output1:",model1(input1).shape)

# 写法2：
model2=Model2()
print("model2:",model2)
input2=torch.ones((64,3,32,32))
print("output2:",model2(input2).shape)
