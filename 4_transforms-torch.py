# transforms:对输入图片进行变换。主要指transforms.py,
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2





#Totensor:转化为Tensor类型
img_path="flower-photos/daisy/11642632_1e7627a2cc.jpg"
img=Image.open(img_path)
print(img)
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
print("tensor_img:",tensor_img)

# Normalize：归一化。
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(tensor_img)
print("img_norm:",img_norm)

# Resize(),改变图像大小。
trans_resize=transforms.Resize((300,300))
img_resize=trans_resize(img)
print("img_resize:",img_resize)

# compose:Compose类是PyTorch的torchvision库中transforms模块的一个重要组成部分
# ，它允许我们将多个transform操作串联起来，形成一个完整的预处理流程.
trans_resize_2=transforms.Resize((512,512))
trans_compose=transforms.Compose([trans_resize_2,tensor_trans])
img_resize_2=trans_compose(img)
print("img_resize_2:",img_resize_2.shape,img_resize_2)
