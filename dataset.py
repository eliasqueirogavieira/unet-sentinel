# import the necessary packages
from lib2to3.pytree import convert
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from osgeo import gdal
import torch
from PIL import Image
import numpy as np
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms=None):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		#image = cv2.imread(imagePath)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		im1 = gdal.Open(imagePath)
		arr = []
		for i in range(1, 3):
			arr.append(im1.GetRasterBand(i).ReadAsArray())
		im1 = None
		image = (np.array(arr)/255).astype(np.float32)
		#im2 = Image.fromarray(np.uint8(im2 * 255))
		#convert_tensor = transforms.ToTensor()
		#image = convert_tensor(im2)
		mask = cv2.imread(self.maskPaths[idx], 0)
		mask = (mask / mask.max()).astype(np.float32)
		#mask = convert_tensor(mask)
		# return a tuple of the image and its mask
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = image.transpose(1, 2, 0)
			transformed = self.transforms(image=image, mask=mask)
			image = transformed["image"]
			mask = transformed["mask"]
		return (image, mask)
