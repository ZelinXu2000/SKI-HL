import logging
import os
import time
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from unet import UNet
from utils.data_loading import ImgDataset
from utils.utils import *


def train_model(
		model,
		device,
		image_path,
		mask_path,
		N,
		patch_size,
		stride,
		epochs,
		batch_size,
		learning_rate
):
	# 1. Create dataset
	dataset = ImgDataset(image_path, mask_path, patch_size, stride, N)

	# 2. Split into train / validation partitions
	val_percent = 0.2
	n_val = int(len(dataset) * val_percent)
	n_train = len(dataset) - n_val
	train_set, valid_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

	# 3. Create data loaders
	loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
	train_loader = DataLoader(train_set, shuffle=True, **loader_args)
	valid_loader = DataLoader(valid_set, shuffle=False, drop_last=True, **loader_args)

	# (Initialize logging)
	logging.info(f'''Starting training:
		Epochs:          {epochs}
		Batch size:      {batch_size}
		Learning rate:   {learning_rate}
		Device:          {device.type}
		Patch size:		 {patch_size}
	''')

	# 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
	optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
	grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
	criterion = nn.BCEWithLogitsLoss()
	global_step = 0
	early_stop = EarlyStopping(patience=10)

	# 5. Begin training
	# pooling = nn.MaxPool2d(N, N)
	pooling = nn.AvgPool2d(N, N)

	start = time.time()
	for epoch in range(1, epochs + 1):
		model.train()
		epoch_loss = 0

		num_train_batches = len(train_loader)
		acc = 0
		with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
			for batch in train_loader:
				images, true_masks = batch['image'], batch['mask']
				images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
				true_masks = true_masks.to(device=device, dtype=torch.float32)

				with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
					masks_pred = model(images)
					masks_pred = pooling(masks_pred)
					loss = criterion(masks_pred, true_masks.float())

					true_masks = (true_masks > 0.5).float()
					masks_pred = (masks_pred > 0.5).float()
					acc += torch.sum(masks_pred == true_masks) / torch.numel(masks_pred)

				optimizer.zero_grad(set_to_none=True)
				grad_scaler.scale(loss).backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				grad_scaler.step(optimizer)
				grad_scaler.update()

				pbar.update(images.shape[0])
				global_step += 1
				epoch_loss += loss.item()
				pbar.set_postfix(**{'loss (batch)': loss.item()})

		epoch_loss = epoch_loss / num_train_batches
		scheduler.step(epoch_loss)
		acc = acc / num_train_batches

		logging.info('Epoch %d: Training Loss %f, Training Acc %f' % (epoch, epoch_loss, acc))
		print('Epoch %d: Training Loss %f, Training Acc %f' % (epoch, epoch_loss, acc))
		logging.info('Current learning rate: {}'.format(scheduler._last_lr))

		# evaluate
		model.eval()
		epoch_loss = 0
		num_valid_batches = len(valid_loader)
		acc = 0
		for batch in valid_loader:
			images, true_masks = batch['image'], batch['mask']

			images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
			true_masks = true_masks.to(device=device, dtype=torch.float32)

			with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
				masks_pred = model(images)
				masks_pred = pooling(masks_pred)
				loss = criterion(masks_pred, true_masks.float())

				true_masks = (true_masks > 0.5).float()
				masks_pred = (masks_pred > 0.5).float()
				acc += torch.sum(masks_pred == true_masks) / torch.numel(masks_pred)

			epoch_loss += loss.item()

		epoch_loss = epoch_loss / num_valid_batches
		acc = acc / num_valid_batches
		logging.info('Epoch %d: Validation Loss %f, Validation Acc %f' % (epoch, epoch_loss, acc))
		print('Epoch %d: Validation Loss %f, Validation Acc %f' % (epoch, epoch_loss, acc))

		early_stop(epoch_loss)
		if early_stop.early_stop:
			logging.info('Early stop!')
			print('Early stop!')
			break

	end = time.time()
	logging.info(f"Deep learning module training time: {(end - start):.3f} seconds")


def dl_train(dataset_id, N, mask_path, epochs, batch_size, lr, patch_size, stride, device, save_id=''):
	logging.info(f'----------------------Deep learning part START----------------------')
	print(f'----------------------Deep learning part START----------------------')

	model = UNet(n_channels=3, n_classes=1)
	model.to(device=device)

	image_path = 'data/dataset' + str(dataset_id) + '/image.tif'

	train_model(model, device, image_path, mask_path, N, patch_size, stride, epochs, batch_size, lr)

	path = 'output/dataset' + str(dataset_id) + save_id + '/model_para'
	if not os.path.exists(path):
		os.mkdir(path)
	torch.save(model, path + '/model_' + str(N) + '.pth')

	logging.info(f'----------------------Deep learning part END----------------------')
	print(f'----------------------Deep learning part END----------------------')
