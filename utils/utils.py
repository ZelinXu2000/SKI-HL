from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import numpy as np


def uncertainty(p):
	p = np.clip(p, 1e-5, 1 - 1e-5)
	return - p * np.log(p) - (1 - p) * np.log(1 - p)


def uncertain_mapping(width, eta, uncertain_label_path, uncertain_threshold, map_value=None, desc='uncertain'):
	inferred_label = Image.open(uncertain_label_path)
	inferred_label = np.array(inferred_label)
	uncertainty_map = uncertainty(inferred_label)
	uncertain_area = uncertainty_map > uncertain_threshold

	target2pixels = dict()
	area2value = dict()
	pixel2target = dict()

	if map_value is not None:
		value_f = map_value.flatten()

	for cur_id in tqdm(range(uncertain_area.size), desc=desc):
		row = cur_id // (width // eta)  # in coarse res
		column = cur_id % (width // eta)
		new_row = row * eta  # in fine res
		new_column = column * eta
		new_id = new_row * width + new_column

		if not uncertain_area[row, column]:  # certain
			temp = []
			for i in range(eta):
				temp += list(range(new_id + i * width, new_id + i * width + eta))
			target2pixels[new_id] = temp
			for t in temp:
				pixel2target[t] = new_id

			if map_value is not None:
				area2value[new_id] = np.mean(value_f[temp])

		else:  # uncertain
			for i in range(eta):
				for j in range(eta):
					temp = new_id + i * width + j
					target2pixels[temp] = [temp]
					pixel2target[temp] = temp

					if map_value is not None:
						area2value[temp] = map_value[new_row + i][new_column + j]

	return target2pixels, area2value, pixel2target


def output_img(model_path, img_path, patch_size, device):
	image = Image.open(img_path)
	image = transforms.ToTensor()(image)

	model = torch.load(model_path, map_location=device)
	W = image.shape[2]
	image_patches = image.data.unfold(0, 3, 3).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
	image_patches = image_patches.to(device)
	concat = np.random.rand(1, W)
	concat_label = np.random.rand(1, W)
	for x in tqdm(range(image_patches.shape[1]), desc='Concat prediction'):
		patch = image_patches[0][x][0].unsqueeze(0)
		pred = model(patch).squeeze(0).squeeze(0)
		concat_row = pred.cpu().detach().numpy()
		pred_label = (torch.sigmoid(pred) > 0.5).float()
		concat_row_label = pred_label.cpu().detach().numpy()
		for y in range(1, image_patches.shape[2]):
			patch = image_patches[0][x][y].unsqueeze(0)
			pred = model(patch).squeeze(0).squeeze(0)

			concat_row = np.concatenate((concat_row, pred.cpu().detach().numpy()), 1)
			pred_label = (torch.sigmoid(pred) > 0.5).float()
			concat_row_label = np.concatenate((concat_row_label, pred_label.cpu().detach().numpy()), 1)
		concat = np.concatenate((concat, concat_row))
		concat_label = np.concatenate((concat_label, concat_row_label))

	return concat[1:, :], concat_label[1:, :]


class EarlyStopping:
	"""
	Early stopping to stop the training when the loss does not improve after
	certain epochs.
	"""

	def __init__(self, patience=5, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is
			   not improving
		:param min_delta: minimum difference between new loss and old loss for
			   new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			# reset counter if validation loss improves
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			# logging.info(f"Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				# logging.info('Early stop!')
				self.early_stop = True
