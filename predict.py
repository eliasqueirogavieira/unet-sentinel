# USAGE
# python predict.py
# import the necessary packages
import config
import cv2
import numpy as np
import torch
from pathlib import Path
import csv
from pathlib import Path
import utils

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
image_paths_list = utils.get_folder_images(config.TEST_IMAGES_FOLDER)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
f = open(Path(config.IMAGES_OUTPUT) / 'stats.csv', 'w')
# iterate over the randomly selected test image paths
writer = csv.writer(f)
writer.writerow(['name','precision','recall','f1 score'])
avg_precision = 0 
avg_recall = 0
avg_f1score = 0

for image_path in image_paths_list:
	print(image_path)
	gtmask_path = Path(config.TEST_MASKS_FOLDER) / Path(image_path).name
	gtmask_path = str(Path(gtmask_path).with_suffix('.png'))
	image_torch, mask_torch, pred_torch = utils.predict_from_paths(unet, image_path, gtmask_path, apply_TTA=config.APPLY_TTA)
	precision, recall, f1 = utils.get_metrics(pred_torch, mask_torch, apply_sigmoid=False)
	precision_np = precision.cpu().numpy()
	recall_np = recall.cpu().numpy()
	f1score_np = f1.cpu().numpy()
	avg_precision += precision_np
	avg_recall += recall_np
	avg_f1score += f1score_np
	pred_torch_bin = pred_torch > config.THRESHOLD
	pred_torch_bin_np = (pred_torch_bin*255).cpu().numpy()
	file_name = str(Path(config.IMAGES_OUTPUT) / Path(image_path).with_suffix('.png').name)
	cv2.imwrite(file_name, pred_torch_bin_np)
	#utils.save_torch_plot(image_torch, mask_torch, pred_torch_bin, file_name)
	writer.writerow([Path(image_path).name, precision_np, recall_np, f1score_np])
avg_precision /= len(image_paths_list)
avg_recall /= len(image_paths_list)
avg_f1score /= len(image_paths_list)
writer.writerow(['Average', avg_precision, avg_recall, avg_f1score])
f.close()