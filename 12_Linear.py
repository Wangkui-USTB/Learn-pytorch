import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
input = torch.randn(128, 20)
m = nn.Linear(20, 30)
output = m(input)
print(output.size())

dataset=torchvision.datasets.CIFAR10("dataset",transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=16)
class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.liner=nn.Linear(49152,10)
	def forward(self,x):
		x=self.liner(x)
		return x
wk=Model()
for data in dataloader:
	images,labels=data
	print("images.shape:",images.shape)
	output=torch.reshape(images,(1,1,1,-1))
	print("output.shape:",output.shape)
	OUTPUT=wk(output)
	print("OUTPUT.shape:", OUTPUT.shape)