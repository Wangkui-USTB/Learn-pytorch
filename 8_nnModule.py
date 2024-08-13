import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.conv1=nn.Conv2d(1,30,5)
		self.conv2=nn.Conv2d(30,1,5)
	def forward(self,x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return x
