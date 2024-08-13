# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
# return_indices=False, ceil_mode=False)[source

# Parameters
# kernel_size (Union[int, Tuple[int, int]]) – the size of the window to take a max over
#
# stride (Union[int, Tuple[int, int]]) – the stride of the window. Default value is kernel_size
#
# padding (Union[int, Tuple[int, int]]) – Implicit negative infinity padding to be added on both sides
#
# dilation (Union[int, Tuple[int, int]]) – a parameter that controls the stride of elements in the window
#
# return_indices (bool) – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
#
# ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape
import torch
from torch import nn
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,3,1],
                    [5,8,0,9,1]],dtype=torch.float32)
input=torch.reshape(input,(-1,1,5,5))
print("input.shape:",input.shape)
class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.maxpooling=nn.MaxPool2d(3,ceil_mode=True)
	def forward(self,x):
		x=self.maxpooling(x)
		return x
wk=Model()
print("maxpooling",wk(input),wk(input).shape)
