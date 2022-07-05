# import the necessary packages
from sys import path
import torch
import os
# base path of the dataset
DATASET_PATH = "/home/eliasqueiroga/Documents/unet/dataset/train"
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images3")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks3")

# define the test split
TEST_SPLIT = 0.001
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 4
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 2000
BATCH_SIZE = 64
# define the input image dimensions
INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 640
# define threshold to filter weak predictions
THRESHOLD = 0.9
# define the path to the base output directory
BASE_OUTPUT = "/home/eliasqueiroga/Documents/unet/output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.pth") #0.5 onefactor
MODEL2_PATH = os.path.join(BASE_OUTPUT, "unet2.pth") #0.33 onefactor
MODEL3_PATH = os.path.join(BASE_OUTPUT, "unet3.pth") #0.35 onefactor
MODEL4_PATH = os.path.join(BASE_OUTPUT, "unet4.pth") #0.37
MODEL5_PATH = os.path.join(BASE_OUTPUT, "unet5.pth") #0.35 com mais augmentation
MODEL6_PATH = os.path.join(BASE_OUTPUT, "unet6.pth") #1024x1024 2000 epocas lr 0.0001
MODEL7_PATH = os.path.join(BASE_OUTPUT, "unet7.pth") #640x640 lr 0.0005
MODEL8_PATH = os.path.join(BASE_OUTPUT, "unet8.pth") #640x640 lr 0.0001 p0.2 of 0.35
MODEL9_PATH = os.path.join(BASE_OUTPUT, "unet9.pth") #640x640 lr 0.001
MODEL10_PATH = os.path.join(BASE_OUTPUT, "unet10.pth") #640x640 lr 0.0001 p0.3
MODEL11_PATH = os.path.join(BASE_OUTPUT, "unet11.pth") #640x640 lr 0.0001 of 0.34
MODEL12_PATH = os.path.join(BASE_OUTPUT, "unet12.pth") #640x640 lr 0.0001 of 0.36
MODEL13_PATH = os.path.join(BASE_OUTPUT, "unet13.pth") #640x640 lr 0.0001 of 0.33
MODEL14_PATH = os.path.join(BASE_OUTPUT, "unet14.pth") #640x640 lr 0.0001 of 0.37
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])