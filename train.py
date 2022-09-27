# USAGE
# python train.py
# import the necessary packages
from dataset import SegmentationDataset
from model import UNet
from utils import *
import config
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from imutils import paths
from tqdm import tqdm
import glob
import torch
import logging
import time
import re
from pathlib import Path
from augmentations import build_augmentations
from sklearn.model_selection import train_test_split
import pytorch_model_summary

logging.basicConfig(filename='treino.log', encoding='utf-8', level=logging.DEBUG)

trainImages = sorted(list(paths.list_images(config.TRAIN_IMAGES_FOLDER)))
trainMasks = sorted(list(paths.list_images(config.TRAIN_MASKS_FOLDER)))
validationImages = sorted(list(paths.list_images(config.VAL_IMAGES_FOLDER)))
validationMasks = sorted(list(paths.list_images(config.VAL_MASKS_FOLDER)))

# # load the image and mask filepaths in a sorted manner
# imagePaths = sorted(list(paths.list_images(config.ALL_IMAGES_PATH)))
# maskPaths = sorted(list(paths.list_images(config.ALL_MASKS_PATH)))
# # partition the data into training and validation splits using 80% of
# # the data for training and the remaining 20% for validate
# split = train_test_split(imagePaths, maskPaths,
# 	test_size=config.VALIDATION_SPLIT, random_state=42)
# # unpack the data split
# (trainImages, validationImages) = split[:2]
# (trainMasks, validationMasks) = split[2:]
# # write the validation image paths to disk so that we can use then
# # when evaluating/validate our model
# print("[INFO] saving validation image paths...")
# f = open(config.VALIDATION_PATHS, "w")
# f.write("\n".join(validationImages))
# f.close()

# define transformations

transforms = build_augmentations()

# create the train and validation datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
validationDS = SegmentationDataset(imagePaths=validationImages, maskPaths=validationMasks, transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(validationDS)} examples in the validation set...")
# create the training and validation data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=config.NUM_THREADS)
validationLoader = DataLoader(validationDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=config.NUM_THREADS)

# calculate steps per epoch for training and validation set
train_steps = len(trainDS) // config.BATCH_SIZE
validation_steps = len(validationDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": []}
logging.info('Novo treinamento')	
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
writer = SummaryWriter(config.BASE_OUTPUT)
cont_iter = 0
cont_val_iter = 0

all_checkpoints = glob.glob(str(config.CHECKPOINT_PATH) + '/*')
if len(all_checkpoints):
	sort_checkpoints = sorted(all_checkpoints, reverse=True)
	last_checkpoint = sort_checkpoints[0]
	unet = torch.load(last_checkpoint)
	last_epoch = int(re.findall(r'\d+', last_checkpoint)[0])
else:
	unet = UNet(encChannels=config.ENC_CHANNELS,decChannels=config.DEC_CHANNELS).to(config.DEVICE)
	last_epoch = 0

#opt = Adam(unet.parameters(), lr=config.INIT_LR)
opt = AdamW(unet.parameters(), lr=config.INIT_LR)

if Path(config.OPTIMIZER_FILE).exists():
	opt = opt.load_state_dict(config.OPTIMIZER_FILE)

config.CHECKPOINT_PATH.mkdir(exist_ok=True)
early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=True)
print(pytorch_model_summary.summary(UNet(encChannels=config.ENC_CHANNELS, decChannels=config.DEC_CHANNELS), torch.zeros(1, 2, 512, 512)))

for e in tqdm(range(config.NUM_EPOCHS), initial=last_epoch):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	total_train_loss = 0
	total_val_loss = 0
	total_val_precision = 0
	total_train_recall = 0
	total_train_precision = 0
	total_val_recall = 0
	total_train_f1 = 0
	total_val_f1 = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = bce_with_logits_loss(pred, y)
		#loss = dynamic_bce(pred, y)
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
		precision, recall, f1 = get_metrics(pred, y)
		total_train_precision += precision
		total_train_recall += recall
		total_train_f1 += f1
		# add the loss to the total training loss so far
		if cont_iter % 50 == 0:
			log_tsb_scalars(writer, 'train/iteration', loss, precision, recall, f1, cont_iter)
			log_tsb_images(writer, 'train/images', torch.sigmoid(pred[0]), y[0], cont_iter)			
		cont_iter += 1
		total_train_loss += loss
	# switch off autograd
	if (e+1) % config.CHECKPOINT_INTERVAL == 0:
		torch.save(unet, str(config.CHECKPOINT_FILE).format(e))
		optimizer_state_dict = opt.state_dict()
		torch.save(optimizer_state_dict, config.OPTIMIZER_FILE)

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
		for (x, y) in validationLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x)
			loss = bce_with_logits_loss(pred, y)
			#loss = dynamic_bce(pred, y)
			precision, recall, f1 = get_metrics(pred, y)
			total_val_precision += precision
			total_val_recall += recall
			total_val_f1 += f1
			if cont_val_iter % 50 == 0:
				log_tsb_scalars(writer, 'val/iteration', loss, precision, recall, f1, cont_val_iter)
				log_tsb_images(writer, 'val/images', torch.sigmoid(pred[0]), y[0], cont_val_iter)			
			cont_val_iter += 1
			total_val_loss += loss


	# calculate the average training and validation loss
	avg_train_loss = total_train_loss / train_steps
	avg_train_precision = total_train_precision / train_steps
	avg_train_recall = total_train_recall / train_steps
	avg_train_f1 = total_train_f1 / train_steps
	log_tsb_scalars(writer, 'train/epoch', avg_train_loss, avg_train_precision, avg_train_recall, avg_train_f1, e)
	
	avg_val_loss = total_val_loss / validation_steps
	avg_val_precision = total_val_precision / validation_steps
	avg_val_recall = total_val_recall / validation_steps
	avg_val_f1 = total_val_f1 / validation_steps
	log_tsb_scalars(writer, 'val/epoch', avg_val_loss, avg_val_precision, avg_val_recall, avg_val_f1, e)
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
		avg_train_loss, avg_val_loss))
	early_stopping(avg_val_loss, unet)
        
	if early_stopping.early_stop:
		print("Early stopping")
		break
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)