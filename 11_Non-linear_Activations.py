import torch
from torch import nn
input=torch.tensor([[1,-0.5],[2.1,5]])
input=torch.reshape(input,(-1,1,2,2))
print("input.shape:",input.shape)
class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.relu=nn.ReLU()
	def forward(self,x):
		x=self.relu(x)
		return x
wk=Model()
output=wk(input)
print("output:",output)
print("input:",input)