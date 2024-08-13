import torch
from PIL import Image
import torchvision
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_path="dog.png"
image=Image.open(image_path).convert('RGB')
print(image)
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
image=transform(image)
image=image.reshape(-1,3,32,32)
print("image.shape:",image.shape)
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
	
model=torch.load("train_save_model/model_CIFAR10_30.pth").to(device)
output=model(image.to(device))
print("output:",output.argmax(1))