# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
# padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)[source]

# Parameters
# in_channels (int) – Number of channels in the input image
#
# out_channels (int) – Number of channels produced by the convolution
#
# kernel_size (int or tuple) – Size of the convolving kernel
#
# stride (int or tuple, optional) – Stride of the convolution. Default: 1
#
# padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
#
# padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
#
# dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
#
# groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
#
# bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
dataset=torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader=DataLoader(dataset,batch_size=64)
class Model(torch.nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.conv1=nn.Conv2d(3,5,3,1)
	def forward(self,x):
		x=self.conv1(x)
		return x
wk=Model()
for data in dataloader:
	images,labels=data
	print("images.shape:", images.shape)
	output=wk(images)
	print("output.shape:",output.shape)
