import os
import time
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import logging
import torch
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_loading import PretrainDataset
from utils.utils import EarlyStopping


def train_model(
		model,
		device,
		train_set,
		valid_set,
		epochs,
		batch_size,
		learning_rate
):
	# Create data loaders
	loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
	train_loader = DataLoader(train_set, shuffle=True, **loader_args)
	valid_loader = DataLoader(valid_set, shuffle=False, drop_last=True, batch_size=1, num_workers=2, pin_memory=True)

	# (Initialize logging)
	logging.info(f'''Starting training:
		Epochs:          {epochs}
		Batch size:      {batch_size}
		Learning rate:   {learning_rate}
		Device:          {device.type}
	''')

	# Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
	optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
	grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
	criterion = nn.BCEWithLogitsLoss()
	early_stop = EarlyStopping()
	# Begin training
	start = time.time()
	for epoch in range(1, epochs + 1):
		# train
		model.train()
		epoch_loss = 0
		num_train_batches = len(train_loader)
		with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
			for batch in train_loader:
				images, true_masks = batch['image'], batch['mask']
				images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
				true_masks = true_masks.to(device=device, dtype=torch.float32)

				with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
					masks_pred = model(images).squeeze(1)
					loss = criterion(masks_pred, true_masks.float())

				optimizer.zero_grad(set_to_none=True)
				grad_scaler.scale(loss).backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				grad_scaler.step(optimizer)
				grad_scaler.update()

				pbar.update(images.shape[0])
				epoch_loss += loss.item()
				pbar.set_postfix(**{'loss (batch)': loss.item()})

		epoch_loss = epoch_loss / num_train_batches
		scheduler.step(epoch_loss)
		logging.info('Epoch %d, Training loss %f' % (epoch, epoch_loss))
		print('Epoch %d, Training loss %f' % (epoch, epoch_loss))
		logging.info('Current learning rate: {}'.format(scheduler._last_lr))

		# evaluate
		model.eval()
		epoch_loss = 0
		num_valid_batches = len(valid_loader)
		for batch in valid_loader:
			images, true_masks = batch['image'], batch['mask']
			images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
			true_masks = true_masks.to(device=device, dtype=torch.float32)

			with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
				masks_pred = model(images).squeeze(1)
				loss = criterion(masks_pred, true_masks.float())

			epoch_loss += loss.item()

		epoch_loss = epoch_loss / num_valid_batches
		logging.info('Epoch %d, Validation loss %f' % (epoch, epoch_loss))
		print('Epoch %d, Validation loss %f' % (epoch, epoch_loss))

		if epoch > 50:
			early_stop(epoch_loss)
			if early_stop.early_stop:
				logging.info('Early stop!')
				break

	end = time.time()
	logging.info(f"Pretrain time: {(end - start):.3f} seconds")


def pretrain(dataset_id, patch_size, seed_water, seed_dry, device, c=''):
	# hyperparameter
	epochs = 200
	batch_size = 4
	lr = 1e-6

	# Create model
	model = UNet(n_channels=3, n_classes=1, dropout=0)
	model.to(device=device)

	# Pretrain
	image_path = 'data/dataset' + str(dataset_id) + '/image.tif'
	image = Image.open(image_path)
	image = transforms.ToTensor()(image)
	height, width = image.shape[1] // patch_size, image.shape[2] // patch_size
	mask_patches = torch.zeros([height * width, patch_size, patch_size])

	with torch.no_grad():
		for i in seed_water:
			mask_patches[i] = 1
		for i in seed_dry:
			mask_patches[i] = 0
	dataset = PretrainDataset(image_path, mask_patches, patch_size)

	train_size = len(seed_water) // 2
	train_set_w, val_set_w = Subset(dataset, seed_water[:train_size]), Subset(dataset, seed_water[train_size:])
	train_set_d, val_set_d = Subset(dataset, seed_dry[:train_size]), Subset(dataset, seed_dry[train_size:])
	train_set = ConcatDataset([train_set_w, train_set_d])
	val_set = ConcatDataset([val_set_w, val_set_d])

	train_model(model, device, train_set, val_set, epochs, batch_size, lr)
	path = 'output/dataset' + str(dataset_id) + c + '/model_para'
	if not os.path.exists(path):
		os.mkdir(path)
	pretrain_model_path = path + '/pretrain.pth'
	torch.save(model, pretrain_model_path)
