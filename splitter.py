from pycocotools.coco import COCO
import os
import os.path
from PIL import Image
import numpy as np
from osgeo import gdal
from pathlib import Path
from matplotlib import pyplot as plt
coco = COCO("/home/eliasqueiroga/Documents/unet/dataset/test.json")
save_folder = "/home/eliasqueiroga/Documents/unet/dataset/test/masks3/"

for j in range(1,101):
    image_id = j
    img = coco.imgs[image_id]
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    image_name = img['file_name']
    image_name = image_name.split('.')[0]
    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    mask = Image.fromarray(np.uint8(mask * 255))
    mask.save(save_folder + image_name + '.png')
path = Path("/home/eliasqueiroga/Documents/unet/dataset/test/images3/") 
for x in path.iterdir():
    print(x)
#arr = os.listdir("/home/eliasqueiroga/Documents/unet/dataset/test/images3/")
#print(arr)