# coding=utf-8
import torch
from torchvision import datasets,transforms
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
# 定义网络结构
class Models(torch.nn.Module):
	def __init__(self):
		super(Models,self).__init__()
		self.connect1=nn.Linear(784,256)
		self.connect2=nn.Linear(256,64)
		self.connect3=nn.Linear(64,10)
		self.softmax=nn.LogSoftmax(dim=1)
		self.relu=nn.ReLU()
	def forward(self,x):
		x=self.connect1(x)
		x=self.relu(x)
		x=self.connect2(x)
		x=self.relu(x)
		x=self.connect3(x)
		x=self.softmax(x)
		return x
class Test:
	def __init__(self):
		self.epoch=5
		self.batch_size=6
		self.learning_rate=0.005
		self.models=Models()
	def loaddata(self):
		dataset=datasets.MNIST("mnist_data",download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]))
		dataset=torch.utils.data.DataLoader(dataset,self.batch_size)
		return dataset
	def lossfunction(self):
		criterion=nn.NLLLoss()
		return criterion
def main(datahandle,models):
	dataset=datahandle().loaddata()
	models=models()
	
	criterion=datahandle().lossfunction()
	optimizer=optim.SGD(models.parameters(),datahandle().learning_rate)
	epoch=datahandle().epoch
	
	for singe_epoch in range(epoch):
		running_loss=0
		for image,label in dataset:
			image=image.view(image.shape[0],-1)
			optimizer.zero_grad()
			output=models(image)
			loss=criterion(output,label)
			loss.backward()
			optimizer.step()
			running_loss+=loss.item()
		print(f"第{singe_epoch}代，训练损失：{running_loss/len(dataset)}")




if __name__=='__main__':
	main(Test,Models)
	