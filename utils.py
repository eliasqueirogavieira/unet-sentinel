import config
from torch.nn import BCEWithLogitsLoss
import torch
from torchmetrics.functional import precision_recall, f1_score
import numpy as np
from osgeo import gdal
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import albumentations as A

def get_metrics(pred, y, apply_sigmoid=True):
	if apply_sigmoid:
		pred = torch.sigmoid(pred)
	pred = pred > config.THRESHOLD
	linear_y = torch.reshape(y>0, (-1,)).type(torch.int)
	linear_pred = torch.reshape(pred, (-1,)).type(torch.float)
	precision, recall = precision_recall(linear_pred, linear_y, average='micro')
	f1_score_val = f1_score(linear_pred,linear_y,average='micro')
	return precision, recall, f1_score_val

def dynamic_bce(pred, y):
	pred_linear = torch.reshape(pred, (-1, ))
	y_linear = torch.reshape(y, (-1, ))
	positive_indexes = torch.where(y_linear==1)[0]
	num_positive = positive_indexes.shape[0]
	num_negative = pred_linear.shape[0] - num_positive
	ones_weight = num_negative*config.ONE_FACTOR/(num_positive+1)
	#logging.debug('Peso: %f', ones_weight)
	pos_weight = torch.ones_like(y_linear)
	pos_weight[positive_indexes] = ones_weight
	loss_func = BCEWithLogitsLoss(pos_weight=pos_weight)
	loss = loss_func(pred_linear, y_linear)
	return loss

def bce_with_logits_loss(pred, y):
	pred_linear = torch.reshape(pred, (-1, ))
	y_linear = torch.reshape(y, (-1, ))
	loss_func = BCEWithLogitsLoss()
	loss = loss_func(pred_linear, y_linear)
	return loss

def log_tsb_scalars(writer, scope, loss, precision, recall, f1, steps):
	writer.add_scalar(scope + '/loss', loss, steps)
	writer.add_scalar(scope + '/precision', precision, steps)
	writer.add_scalar(scope + '/recall', recall, steps)
	writer.add_scalar(scope + '/f1', f1, steps)

def log_tsb_images(writer, scope, pred, y, steps):
    pred_bin = pred > config.THRESHOLD
    pred_heatmap = pred * pred_bin
    writer.add_image(scope + '/ground_mask', y, steps, dataformats='HW')
    writer.add_image(scope + '/pred_mask', pred, steps, dataformats='CHW')
    writer.add_image(scope + '/pred_mask_bin', pred_bin, steps, dataformats='CHW')
    writer.add_image(scope + '/pred_mask_hm', pred_heatmap, steps, dataformats='CHW')

	
def save_torch_plot(orig_image, gt_mask, pred_mask, path):
	# initialize our figure
    orig_image = np.clip(orig_image[0].cpu().numpy(), 0, 1)
    orig_image = orig_image.transpose(1, 2, 0)
    gt_mask = gt_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    new_img = np.concatenate([orig_image[..., 2:], orig_image[..., -1:]], axis=-1)
    ax[0].imshow(new_img)
    ax[1].imshow(gt_mask)
    ax[2].imshow(pred_mask)
    # set the titles of the subplots
    ax[0].set_title("Imagem")
    ax[1].set_title("Máscara original")
    ax[2].set_title("Máscara gerada")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig(path)
    plt.close()

def load_images(image_path, mask_path):
    img_orig = gdal.Open(image_path)
    arr = []
    for i in range(1, 3):
        arr.append(img_orig.GetRasterBand(i).ReadAsArray())
    img_orig = None
    img_append =  (np.array(arr)/255).astype(np.float32)
    gt_mask = cv2.imread(mask_path, 0)
    gt_mask = (gt_mask / gt_mask.max()).astype(np.float32)
    image = np.expand_dims(img_append, 0)
    return image, gt_mask

def augment_image(image):
    h_transform = A.HorizontalFlip(p=1)
    v_transform = A.VerticalFlip(p=1)
    h_dict = h_transform(image=image)
    v_dict = v_transform(image=image)
    h_image = h_dict['image']
    v_image = v_dict['image']
    return h_image, v_image

def get_mean_mask(stacked_mask):
    h_transform = A.HorizontalFlip(p=1)
    v_transform = A.VerticalFlip(p=1)
    orig_mask = stacked_mask[0]
    h_mask = stacked_mask[1]
    v_mask = stacked_mask[2]
    h_dict = h_transform(image=h_mask.cpu().numpy())
    v_dict = v_transform(image=v_mask.cpu().numpy())
    h_mask_flipped = torch.Tensor(h_dict['image']).cuda()
    v_mask_flipped = torch.Tensor(v_dict['image']).cuda()
    mean_mask = (h_mask_flipped + v_mask_flipped + orig_mask)/3
    return mean_mask

def predict_from_paths(model, image_path, mask_path, apply_TTA=True):
    model.eval()
    image, gt_mask = load_images(image_path, mask_path)
    stacked_image = image
    if apply_TTA:
        h_image, v_image = augment_image(image)
        stacked_image = np.concatenate([image, h_image, v_image], axis=0)
    with torch.no_grad():
        image_torch = torch.from_numpy(stacked_image).to(config.DEVICE)
        image_torch = image_torch.type(torch.float).cuda()
        pred_mask = model(image_torch).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        if apply_TTA:
            pred_mask = get_mean_mask(pred_mask)
            image_torch = image_torch[:1]
        gt_mask = torch.from_numpy(gt_mask).to(config.DEVICE)
    return image_torch, gt_mask, pred_mask


def get_folder_images(path_folder_str):
    path = Path(path_folder_str) 
    path_list = list(map(lambda path: str(path), path.iterdir()))
    return path_list

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path=Path(config.BASE_OUTPUT) / 'es_best_model.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss

def count_parameters(model):
    total_params = 0
    for parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    print(f"Total Trainable Params: {total_params}")
    