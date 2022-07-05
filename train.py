# USAGE
# python train.py
# import the necessary packages
from dataset import SegmentationDataset
from model import UNet
import config
import loss
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
#import matplotlib.pyplot as plt
import torch
from torch import nn, threshold
import logging
import time
from torchmetrics.functional import precision_recall
import os
import numpy as np
from pathlib import Path

logging.basicConfig(filename='treino.log', encoding='utf-8', level=logging.DEBUG)
CHECKPOINT_INTERVAL = 50
ONE_FACTOR = 0.34

def dynamic_bce(pred, y):
	pred_linear = torch.reshape(pred, (-1, ))
	y_linear = torch.reshape(y, (-1, ))
	positive_indexes = torch.where(y_linear==1)[0]
	num_positive = positive_indexes.shape[0]
	num_negative = pred_linear.shape[0] - num_positive
	ones_weight = num_negative*ONE_FACTOR/(num_positive+1)
	#logging.debug('Peso: %f', ones_weight)
	pos_weight = torch.ones_like(y_linear)
	pos_weight[positive_indexes] = ones_weight
	lossFunc = BCEWithLogitsLoss(pos_weight=pos_weight)
	loss = lossFunc(pred_linear, y_linear)
	return loss

def dice_loss(pred, y, epsilon=1e-6):        

	#pred = torch.sigmoid(pred)  
	
	pred = pred.view(-1)
	y = y.view(-1)
	
	intersection = (pred * y).sum()                            
	dice = (2.*intersection + epsilon)/(pred.sum() + y.sum() + epsilon)  

	return 1 - dice

def iou_loss(pred, y, epsilon=1e-6):
	
	#comment out if your model contains a sigmoid or equivalent activation layer
	#pred = torch.sigmoid(pred)       
	
	#flatten label and prediction tensors
	pred = pred.view(-1)
	y = y.view(-1)
	
	#intersection is equivalent to True Positive count
	#union is the mutually inclusive area of all labels & predictions 
	intersection = (pred * y).sum()
	total = (pred + y).sum()
	union = total - intersection 
	
	IoU = (intersection + epsilon)/(union + epsilon)
			
	return 1 - IoU

""" class ComboLoss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(ComboLoss, self).__init__()

	def forward(self, inputs, targets, smooth=1, eps=1e-9):
		
		#PyTorch
		ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
		CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)

		#True Positives, False Positives & False Negatives
		intersection = (inputs * targets).sum()    
		dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

		inputs = torch.clamp(inputs, eps, 1.0 - eps)       
		#out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
		out = dynamic_bce(targets, inputs)
		weighted_ce = out.mean(-1)
		combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
		return combo """

# load the image and mask filepaths in a sorted manner
trainPaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
trainMaskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(trainPaths, trainMaskPaths,
	test_size=config.TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# define transformations
transforms = transforms.Compose(
        [transforms.RandomVerticalFlip(p=0.2),
		 transforms.RandomHorizontalFlip(p=0.2)
		])


# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())

# initialize our UNet model
unet = UNet().to(config.DEVICE)
# initialize loss function and optimizer
#pos_weight = torch.ones((512, 512), dtype=torch.float64) * 1.7437
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}
logging.info('Novo treinamento')	
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
writer = SummaryWriter(config.BASE_OUTPUT)
cont_iter = 0
cont_val_iter = 0

checkpoint_path = Path(config.BASE_OUTPUT) / 'checkpoints'
checkpoint_path.mkdir(exist_ok=True)

for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	totalPrecision = 0
	totalRecall = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = dynamic_bce(pred, y)
		if torch.isnan(loss) or torch.isinf(loss):
			print(pred.min().item())
			print(pred.max().item())
			print(y.min().item())
			print(y.max().item())
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		arr_gradmax = []
		arr_gradmin = []
		# add the loss to the total training loss so far
		if cont_iter % 50 == 0:
			writer.add_scalar('train/iteration_loss', loss, cont_iter)
		cont_iter += 1
		totalTrainLoss += loss
	# switch off autograd
	if e % CHECKPOINT_INTERVAL == 0:
		torch.save(unet, config.BASE_OUTPUT + '/checkpoints/epoch{:02d}.pth'.format(e))

	grad_log = list(map(lambda v: v.grad.cpu().numpy(), unet.parameters()))
	for k in range(len(grad_log)):
		arr_gradmax.append(k)
		arr_gradmin.append(k)
		arr_gradmax[k] = grad_log[k].max()
		arr_gradmin[k] = grad_log[k].min()
	arr_max = max(arr_gradmax)
	arr_min = min(arr_gradmin)
	logging.debug('Epoch: %d / Max grad: %f / Min grad: %f', e+1, arr_max, arr_min)	
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x)
			loss = dynamic_bce(pred, y)
			totalTestLoss += loss
			if cont_val_iter % 50 == 0:
				writer.add_scalar('test/iteration_loss', loss, cont_val_iter)
			cont_val_iter += 1


	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	writer.add_scalar('train/epoch_loss', avgTrainLoss, e)
	avgTestLoss = totalTestLoss / testSteps
	writer.add_scalar('test/epoch_loss', avgTestLoss, e)
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
#check_accuracy(testLoader, unet, config.DEVICE)

# plot the training loss
""" plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH) """
# serialize the model to disk
torch.save(unet, config.MODEL_PATH)