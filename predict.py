# USAGE
# python predict.py
# import the necessary packages
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from pathlib import Path
import os
from osgeo import gdal
from torchvision import transforms
from PIL import Image
from torchmetrics.functional import precision_recall
import csv
from pathlib import Path

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	if origImage.max() > 1 or origImage.min() < 0:
		print('Original image is out of bounds')
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Imagem")
	ax[1].set_title("Máscara original")
	ax[2].set_title("Máscara gerada")
	# set the layout of the figure and display it
	figure.tight_layout()
	plt.savefig('figura.png')
	plt.close()


def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		#image = cv2.imread(imagePath)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#image = image.astype("float32") / 255.0
		im1 = gdal.Open(imagePath)
		arr = []
		for i in range(1, 5):
			arr.append(im1.GetRasterBand(i).ReadAsArray())
		im1 = None
		im2 = np.array(arr).transpose(1, 2, 0)
		im3 = np.array(arr)
		#im3 = im3.astype("float32") / 255.0
		#convert_tensor = transforms.ToTensor()
		#image = convert_tensor(im3).to(config.DEVICE)
		# resize the image and make a copy of it for visualization
		#image = cv2.resize(im2, (512, 512))
		orig = im2.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join("/home/eliasqueiroga/Documents/unet/dataset/test/masks3",
			filename)
		groundTruthPath = str(Path(groundTruthPath).with_suffix('.png'))
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))
		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		#image = np.transpose(im2, (2, 0, 1))
		image = np.expand_dims(im3, 0)
		image = torch.from_numpy(image).to(config.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu()
		# filter out the weak predictions and convert them to integers
		#predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask > 0.9
		lineargtMasktorch = torch.reshape(torch.Tensor(gtMask>0), (-1,)).type(torch.int)
		linearPredMask = torch.reshape(predMask, (-1,)).type(torch.float)
		precision, recall = precision_recall(linearPredMask, lineargtMasktorch, average='micro')
		predMask = (predMask*255).numpy()
		prepare_plot(np.clip(orig, 0, 1), gtMask, predMask)
		return precision.numpy(), recall.numpy()

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=100)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
#unet2 = torch.load(config.MODEL2_PATH).to(config.DEVICE)
#unet3 = torch.load(config.MODEL3_PATH).to(config.DEVICE)
#unet4 = torch.load(config.MODEL4_PATH).to(config.DEVICE)
#unet5 = torch.load(config.MODEL5_PATH).to(config.DEVICE)
#unet6 = torch.load(config.MODEL6_PATH).to(config.DEVICE)
#unet7 = torch.load(config.MODEL7_PATH).to(config.DEVICE)
#unet8 = torch.load(config.MODEL8_PATH).to(config.DEVICE)
#unet9 = torch.load(config.MODEL9_PATH).to(config.DEVICE)
#unet10 = torch.load(config.MODEL10_PATH).to(config.DEVICE)
#unet11 = torch.load(config.MODEL11_PATH).to(config.DEVICE)
#unet12 = torch.load(config.MODEL12_PATH).to(config.DEVICE)
#unet13 = torch.load(config.MODEL13_PATH).to(config.DEVICE)
#unet14 = torch.load(config.MODEL14_PATH).to(config.DEVICE)
f = open('teste.csv', 'w')
# iterate over the randomly selected test image paths
writer = csv.writer(f)
writer.writerow(['name','precision','recall'])

for path in imagePaths:
	# make predictions and visualize the results
	#make_predictions(unet, path)
	#make_predictions(unet2, path)
	#make_predictions(unet3, path)
	#make_predictions(unet4, path)
	#make_predictions(unet5, path)
	#make_predictions(unet6, path)
	#make_predictions(unet7, path)
	precision, recall = make_predictions(unet, path)
	writer.writerow([Path(path).name, precision, recall])
	#make_predictions(unet9, path)
	#make_predictions(unet10, path)
	#make_predictions(unet13, path)
	#make_predictions(unet8, path)
	#make_predictions(unet12, path)
	#make_predictions(unet14, path)

f.close()