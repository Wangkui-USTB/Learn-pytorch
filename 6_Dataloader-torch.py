import torchvision.datasets
from torch.utils.data import DataLoader
test_data=torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
image,label=test_data[0]

print("image.shape,label:",image.shape,label)

for data in test_loader:
	images,labels=data
	print("images.shape:",images.shape)
	print("labels:", labels)
	print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")