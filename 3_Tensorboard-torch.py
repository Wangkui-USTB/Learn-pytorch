from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 虚拟命令执行
# tensorboard --logdir=logs --port=6006
writer = SummaryWriter("logs")
image_path="flower-photos/daisy/5547758_eea9edfd54_n.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
writer.add_image("test",img_array,1,dataformats="HWC")

# y=x**2
for i in range(100):
	writer.add_scalar("y=x**2",i**2,i)

print("wk")
writer.close()