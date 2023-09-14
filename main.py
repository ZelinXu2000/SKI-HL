import logging
import os
import time
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

from dl_train import dl_train
from pretrain import pretrain
from psl_inference import inference


def parse_args():
	parser = argparse.ArgumentParser()

	# overall
	parser.add_argument('--save_id', default='', type=str)
	parser.add_argument('--device', default='cuda', type=str)
	parser.add_argument('--dataset', default=1, type=int, help='Dataset ID')
	parser.add_argument('--eta', default=10, type=int, help='Resolution constant')
	parser.add_argument('-K', default=2, type=int, help='Resolution levels')
	parser.add_argument('-s', '--seed_num', default=4, type=int, help='Number of seeds (labels)')
	parser.add_argument('--pretrain', default=1, type=int, help='Pretrain based on sparse labels or not')

	# deep learning
	parser.add_argument('--patch_size', default=100, type=int, help='Patch when split the large image')
	parser.add_argument('--stride', default=100, type=int, help='Stride when split the large image')
	parser.add_argument('--epoch_d', default=100, type=int, help='Epochs for deep learning training')
	parser.add_argument('--batch_size_d', default=4, type=int, help='Batch size for deep learning training')
	parser.add_argument('--lr_d', default=1e-5, type=float, help='Learning rate for deep learning training')

	# logic inference
	parser.add_argument('-p', default=1, type=int, help='Exponent of the distance function')
	parser.add_argument('--epoch_l', default=300, type=int, help='Epochs for logic inference')
	parser.add_argument('--batch_size_l', default=8, type=int, help='Base batch size for logic inference')
	parser.add_argument('--lr_l', default=1e-1, type=float, help='Base learning rate for logic inference')

	return parser.parse_args()


def main():
	# initialize log
	log_path = 'log/'
	if not os.path.exists(log_path):
		os.mkdir(log_path)
	log_path = log_path + str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '.log'
	logging.basicConfig(filename=log_path,
						filemode='a',
						datefmt='%H:%M:%S',
						level=logging.INFO,
						format='%(asctime)s: %(message)s')

	# hyper parameter
	args = parse_args()
	save_id = args.save_id

	dataset_id = args.dataset
	uncertain_thresholds = [0, 0.6, 0.5]

	device = torch.device(args.device)
	logging.info(f'Train on device {device}')
	print(f'Train on device {device}')

	# output path
	output_path = 'output/'
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	path = output_path + 'dataset' + str(dataset_id) + save_id
	if not os.path.exists(path):
		os.mkdir(path)

	# pretrain
	if args.pretrain:
		seed_num = args.seed_num
		if seed_num == 4:  # manually choose 8 seeds (4 for train, 4 for valid)
			if dataset_id == 1:
				seed_water = [329, 226, 365, 194]
				seed_dry = [75, 28, 66, 95]
			elif dataset_id == 2:
				seed_water = [2110, 2150, 1700, 2160]
				seed_dry = [435, 465, 445, 455]
			else:
				raise Exception('No such dataset!')
		else:  # randomly generate seeds
			label = np.asarray(Image.open('data/dataset' + str(dataset_id) + '/label.tif'))
			re_label = cv2.resize(label, (label.shape[1] // args.patch_size, label.shape[0] // args.patch_size),
								  interpolation=cv2.INTER_NEAREST)
			re_label = re_label.flatten()
			seed_water = np.random.choice(np.where(re_label == 1)[0], seed_num)
			seed_dry = np.random.choice(np.where(re_label == 0)[0], seed_num)

		pretrain(dataset_id=dataset_id,
				 patch_size=args.patch_size,
				 seed_water=seed_water,
				 seed_dry=seed_dry,
				 device=device,
				 c=save_id)

	# iteratively update start
	last_infer_path = ''
	model_path = path + '/model_para/pretrain.pth'
	eta = args.eta

	for i in range(args.K, -1, -1):
		N = eta ** i
		logging.info('i = %d, N = %d' % (i, N))
		print('i = %d, N = %d' % (i, N))

		t = uncertain_thresholds[-i - 1]

		# logic inference part
		inference(dataset_id=dataset_id,
				  eta=eta,
				  i=i,
				  model_path=model_path,
				  epoch=args.epoch_l,
				  lr=args.lr_l ** (i + 1),
				  batch_size=args.batch_size_l ** (4 - i),
				  p=args.p,
				  last_infer_path=last_infer_path,
				  uncertain_threshold=t,
				  device=device,
				  save_id=save_id)

		# deep learning part
		mask_path = path + '/label_map/label_map_' + str(N) + '.tif'
		dl_train(dataset_id=dataset_id,
				 N=N,
				 mask_path=mask_path,
				 epochs=args.epoch_d,
				 batch_size=args.batch_size_d,
				 lr=args.lr_d,
				 patch_size=args.patch_size,
				 stride=args.stride,
				 device=device,
				 save_id=save_id)

		# path for next iteration
		last_infer_path = path + '/label_map/label_map_' + str(N) + '.tif'
		model_path = path + '/model_para/model_' + str(N) + '.pth'


if __name__ == '__main__':
	main()
