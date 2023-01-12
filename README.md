
# UNet for classification of SAR Images from Amazon Rainforest

This project consists in a custom version of the UNet architecture, to handle SAR images (Sentinel-1 satellite, multiband spectral images), classifying deforestation using a post-processed dataset with ground truth labeled by Censipam employees using [Planet](https://www.planet.com)
 satellites (optical sensors).


## Requirements
- Python (v3.9.9 was used)
- PyTorch (v1.11 was used)
- Tensorboard (latest version)
- OSGeo GDAL (latest version)
- Also used cv2, albumentations, numpy, torchvision, sklearn, pycocotools, logging, csv and others
## Dataset
The Dataset was composed of clippings from larger TIFF images of different regions from the Legal Amazon Rainforest:
- 1110 training samples, 207 validation samples and 137 test samples.
- All samples had 512x512 resolution. 
- All augmentations used are listed in augmentations.py

All the images were provided by Brazilian Ministry of Defense in a collaboration with Universidade de Brasília (UnB).
## Hyperparameters
- Encoder channels: 2, 16, 32, 64
- Decoder channels: 64, 32, 16
- Learning rate: 1e-3
- Batch size: 16
- Threshold: 0.45
- EarlyStopping Patience: 50 epochs

## Optimizations

Many custom optimizations were implemented and they are located in the utils.py, some of them are listed as follows:
- Full TensorBoard optimization, tracking loss, metrics (F1, recall, precision) and predicted images every 'n' epochs.
- Custom BCE (Binary Cross Entrophy) loss function, to be applied with custom weights for each batch of the dataset. This is optional but it was required to normalize the batches, since many images don`t have deforestation.
- EarlyStopping implemented with the option to resume training if it was interrupted at any point.

## Screenshots

![My Remote Image](https://i.imgur.com/tbB7wpb.png)
![My another remote image](https://i.imgur.com/fGqEx6E.png)


## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/elias-queiroga/)


