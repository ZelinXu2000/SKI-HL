import logging
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FormulaDataset(Dataset):
	def __init__(self, formulas, weights):
		assert len(formulas) == len(weights)
		self.formulas = formulas
		self.weights = weights

	def __getitem__(self, index):
		return self.formulas[index][0], self.formulas[index][1], self.weights[index]

	def __len__(self):
		return len(self.formulas)


class ImgDataset(Dataset):
	def __init__(self, image_path: str, mask_path: str, patch_size, stride, N=1):

		image = Image.open(image_path)
		image = transforms.ToTensor()(image)
		self.image_patches = image.data.unfold(0, 3, 3).unfold(1, patch_size, stride).unfold(2, patch_size, stride)

		mask = Image.open(mask_path)
		mask = transforms.ToTensor()(mask)
		self.mask_patches = mask.data.unfold(0, 1, 1).unfold(1, patch_size // N, stride // N).unfold(2, patch_size // N, stride // N)
		self.dataset_size = self.mask_patches.shape[1] * self.mask_patches.shape[2]
		assert self.dataset_size == self.image_patches.shape[1] * self.image_patches.shape[2], 'Different image and mask numbers'

		logging.info(f'Creating dataset with {self.dataset_size} examples')

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		x = idx // self.image_patches.shape[2]
		y = idx % self.image_patches.shape[2]
		return {
			'image': self.image_patches[0][x][y],
			'mask': self.mask_patches[0][x][y]
		}


class PretrainDataset(Dataset):
	def __init__(self, image_path: str, mask_patches, patch_size):
		image = Image.open(image_path)
		image = transforms.ToTensor()(image)
		self.image_patches = image.data.unfold(0, 3, 3).unfold(1, patch_size, patch_size).unfold(2, patch_size,
																								 patch_size)
		self.image_patches = torch.reshape(self.image_patches, (
			-1, self.image_patches.shape[-3], self.image_patches.shape[-2], self.image_patches.shape[-1]))

		self.mask_patches = mask_patches

		self.dataset_size = self.mask_patches.shape[0]
		assert self.dataset_size == self.image_patches.shape[0], 'Different image and mask numbers'

		logging.info(f'Creating dataset with {self.dataset_size} examples')

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		return {"image": self.image_patches[idx],
				"mask": self.mask_patches[idx]
				}
