import torch
from torch.nn import L1Loss,MSELoss
from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader

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
		x=self.softmax(x)
		return x
# 写法1：
dataset=torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader=DataLoader(dataset,batch_size=64)
model=Model1()
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
epochs=100
sum_epoch=0
for epoch in range(epochs):
	sum_epoch=0
	for data in dataloader:
		optimizer.zero_grad()
		images,labels=data
		output=model(images)
		results=loss(output,labels)
		sum_epoch += results
		results.backward()
		optimizer.step()
	print(f"第{epoch}轮，损失为{sum_epoch/len(dataloader)}")
	