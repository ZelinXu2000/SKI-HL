import os
from torch import nn
import torch.utils.data as data
import logging
import time

from utils.utils import *
from utils.data_loading import FormulaDataset
from utils.hmt import get_tree


class PslLoss(nn.Module):
	def __init__(self):
		super(PslLoss, self).__init__()
		self.relu = nn.ReLU()

	def forward(self, interpretation, formulas_l, formulas_r, weight, p=1):
		d_f = self.relu(interpretation(formulas_l) - interpretation(formulas_r)).to(weight.device)
		l_f = weight.unsqueeze(1) * torch.pow(d_f, p)
		return torch.mean(l_f)


def get_formulas(eta, i, image_shape, dataset_id, last_infer_path='', uncertain_threshold=0.65):
	formulas = []
	weight = []

	height = image_shape[0]
	width = image_shape[1]

	parents, neighbors = get_tree(height, width, eta, i, dataset_id, last_infer_path, uncertain_threshold)

	for i, line in enumerate(tqdm(parents, desc='Loading topology formulas')):
		for p in line[1:]:
			formulas.append([i, p])
			weight.append(0.7)

	for i, line in enumerate(tqdm(neighbors, desc='Loading spatial neighbor formulas')):
		for x in line[1:]:
			formulas.append([i, x])
			weight.append(0.3)

	logging.info('# Variables: %d' % len(parents))
	print('# Variables: %d' % len(parents))
	logging.info('# Formulas: %d' % len(formulas))
	print('# Formulas: %d' % len(formulas))

	return formulas, weight, len(parents)


def initialization(eta, i, image_shape, model_path, image_path, device, last_infer_path='', uncertain_threshold=0.65):
	N = eta ** i
	width = image_shape[1] // N

	# initialize interpretation
	pred_value, pred_label = output_img(model_path, image_path, 100, device)
	pred_value = torch.sigmoid(torch.tensor(pred_value)).unsqueeze(0)
	pooling = nn.AvgPool2d(N, N)

	if len(last_infer_path) != 0:
		pred_value = pooling(pred_value).squeeze(0).numpy()
		_, area2value, _ = uncertain_mapping(width, eta, last_infer_path, uncertain_threshold, pred_value,
											 desc='Initialize')
		para = torch.tensor(list(area2value.values())).unsqueeze(1)
	else:
		para = pooling(pred_value)
		para = para.flatten().unsqueeze(1)
	interpretation = nn.Embedding.from_pretrained(para, freeze=False)

	return interpretation


def optimize(interpretation, data_iter, num_epochs, loss, optimizer, scheduler, early_stop, device, p=1):
	interpretation = interpretation.to(device)
	start = time.time()
	for epoch in range(1, num_epochs + 1):
		l_sum = 0
		num_batch = len(data_iter)
		for batch in tqdm(data_iter, desc='Epoch ' + str(epoch)):
			f_l, f_r, w = [d.to(device) for d in batch]
			l = loss(interpretation, f_l, f_r, w, p)  # loss of this batch
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			l_sum += l.cpu().item()
		l_epoch = l_sum / num_batch
		logging.info('Epoch %d, Loss %f' % (epoch, l_epoch))
		print('Epoch %d, Loss %f' % (epoch, l_epoch))

		early_stop(l_epoch)
		if early_stop.early_stop:
			logging.info('Early stop!')
			print('Early stop!')
			break
		scheduler.step(l_epoch)

	end = time.time()
	logging.info(f"Logic inference module training time: {(end - start):.3f} seconds")


def inference(dataset_id, eta, i, model_path, epoch, lr, batch_size, p, last_infer_path, uncertain_threshold, device,
			  save_id=''):
	logging.info(f'----------------------Logic inference part START----------------------')
	print(f'----------------------Logic inference part START----------------------')

	image_path = 'data/dataset' + str(dataset_id) + '/image.tif'
	image = transforms.ToTensor()(Image.open(image_path))
	image_shape = image.shape[1], image.shape[2]

	path = 'output/dataset' + str(dataset_id) + save_id + '/label_map'
	if not os.path.exists(path):
		os.mkdir(path)

	# ground rules
	formulas, weight, atom_num = get_formulas(eta, i, image_shape, dataset_id, last_infer_path, uncertain_threshold)

	# initialize interpretation
	interpretation = initialization(eta, i, image_shape, model_path, image_path, device, last_infer_path,
									uncertain_threshold)

	# set up
	optimizer = torch.optim.SGD(interpretation.parameters(), lr, 0.99)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
	early_stop = EarlyStopping(min_delta=1e-4)

	l = PslLoss()

	dataset = FormulaDataset(formulas, weight)
	data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

	# start optimize
	optimize(interpretation, data_iter, epoch, l, optimizer, scheduler, early_stop, device, p)

	# store the inferred label map
	N = eta ** i
	height = image_shape[0] // N
	width = image_shape[1] // N
	if len(last_infer_path) != 0:
		target2pixels, _, _ = uncertain_mapping(width, eta, last_infer_path, uncertain_threshold,
												desc='Output inferred label')
		label_map = np.array([-1] * height * width, dtype=np.float32)
		values = interpretation.cpu().weight.data.numpy()
		for i, v in enumerate(list(target2pixels.values())):
			label_map[v] = values[i][0]
		label_map.resize(height, width)
	else:
		label_map = interpretation.cpu().weight.data.resize(height, width).numpy()
	max_v = np.max(label_map)
	min_v = np.min(label_map)
	normalize = (label_map - min_v) / (max_v - min_v)
	Image.fromarray(normalize).save(path + '/label_map_' + str(N) + '.tif')

	logging.info(f'----------------------Logic inference part END----------------------')
	print(f'----------------------Logic inference part END----------------------')
