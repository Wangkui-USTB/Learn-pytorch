import torchvision
import torch
# 保存模型方式1：不仅保存了模型还保存了模型参数。
vgg16=torchvision.models.vgg16(pretrained=False)
torch.save(vgg16,"vgg16_method1.pth")
# 对应的加载模型方法：
model=torch.load("vgg16_method1.pth")
print("model:",model)

# 保存模型方式2：只是保存了模型的参数
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
# 对应的加载模型方法：
model2=torch.load("vgg16_method2.pth")
print("model:",model2)

vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.state_dict(model2)
print(vgg16)
