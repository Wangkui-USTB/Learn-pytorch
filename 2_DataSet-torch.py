#  dataset 为数据编号及提供一种方式获取数据及其标签。1、如何获取每一个数及其标签。2、获取总共有多少数据
#  dataloder 为网络提供不同的新式。
from torch.utils.data import Dataset,ConcatDataset
import cv2
from PIL import Image
import os
class mydata(Dataset):
	def __init__(self,root_dir,label_dir):
		self.root_dir=root_dir
		self.label_dir=label_dir
		self.path=os.path.join(self.root_dir,self.label_dir)
		self.img_path=os.listdir(self.path)
	def __getitem__(self,indx):
		img_name=self.img_path[indx]
		img_item_path=os.path.join(self.path,img_name)
		img=Image.open(img_item_path)
		label=self.label_dir
		return img,label
	def __len__(self):
		return len(self.img_path)
root_dir="flower-photos"
daisy_label_dir="daisy"
dandelion_label_dir="dandelion"
flower_daisy_dataset=mydata(root_dir,daisy_label_dir)
flower_dandelion_dataset=mydata(root_dir,dandelion_label_dir)
train_data=ConcatDataset([flower_daisy_dataset, flower_dandelion_dataset])
imge,label=train_data[1]
print(imge,label)
print("train_data数据长度:",len(train_data))





img_path="flower-photos/daisy/5547758_eea9edfd54_n.jpg"
print(img_path)

dir_path="flower-photos/daisy"
img_path_list=os.listdir(dir_path)
print(img_path_list)
print(img_path_list[0])


root_dir="flower-photos"
label_dir="daisy"
print(os.path.join(root_dir,label_dir))