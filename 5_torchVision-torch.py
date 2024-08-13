import torchvision
dataset_transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root="dataset",train=True,transform=dataset_transforms,download=True)
test_set=torchvision.datasets.CIFAR10(root="dataset",train=False,transform=dataset_transforms,download=True)
print(train_set.classes)
print(train_set[0])
