# import the necessary packages
from pathlib import Path
from sys import path
import torch
import os
# base path of the dataset
ALL_IMAGES_PATH = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg2/all/imgs"
ALL_MASKS_PATH = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg2/all/labels"
TRAIN_IMAGES_FOLDER = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg3/train/images"
TRAIN_MASKS_FOLDER = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg3/train/masks"
TEST_IMAGES_FOLDER = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg3/test/images"
TEST_MASKS_FOLDER = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg3/test/masks"
VAL_IMAGES_FOLDER = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg3/validation/images"
VAL_MASKS_FOLDER = "/home/eliasqueiroga/Documents/unet-sentinel/dataset/seg3/validation/masks"
# define the path to the base output directory
BASE_OUTPUT = "/home/eliasqueiroga/Documents/unet-sentinel/output"
IMAGES_OUTPUT = Path(BASE_OUTPUT) / "preds"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.pth")
VALIDATION_PATHS = os.path.sep.join([BASE_OUTPUT, "validation_paths.txt"])
# define the path to the images and masks dataset
NUM_THREADS = os.cpu_count()
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 2
NUM_CLASSES = 1
NUM_LEVELS = 3
DEC_CHANNELS = (64, 32, 16)
ENC_CHANNELS = (2, 16, 32, 64)
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 10000
BATCH_SIZE = 16
PATIENCE = 50
VALIDATION_SPLIT = 0.2
CHECKPOINT_PATH = Path(BASE_OUTPUT) / 'checkpoints'
num_digits = len(str(NUM_EPOCHS))
CHECKPOINT_FILE = CHECKPOINT_PATH / ('epoch{:0' + str(num_digits) + 'd}.pth')
CHECKPOINT_INTERVAL = 50
OPTIMIZER_FILE = Path(BASE_OUTPUT) / 'optimizer_state.pth'
# define the input image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
# define threshold to filter weak predictions
THRESHOLD = 0.45
ONE_FACTOR = 1
APPLY_TTA = False

