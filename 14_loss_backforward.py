# loss作用：
	# 1：计算实际输出与目标之间的差距。
	# 2：为更新输出提供一定的依据（反向传播）。
import torch
from torch.nn import L1Loss,MSELoss
output=torch.tensor([1,2,3]).float()
targets=torch.tensor([1,2,5]).float()

output=torch.reshape(output,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

loss1=L1Loss()
result1=loss1(output,targets)
print("result1:",result1)

loss2 = MSELoss()
result2=loss2(output,targets)
print("result2:",result2)

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
		return x
# 写法1：
dataset=torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader=DataLoader(dataset,batch_size=64)
model=Model1()
loss=nn.CrossEntropyLoss()
for data in dataloader:
	images,labels=data
	output=model(images)
	results=loss(output,labels)
	results.backward()
	print(results)