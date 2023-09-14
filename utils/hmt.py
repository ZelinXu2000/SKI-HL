from tqdm import tqdm
from PIL import Image
import numpy as np
from torch import nn
import torch

from utils.utils import uncertain_mapping


class Node:
	def __init__(self, value, node_index):
		self.elevation = value
		self.node_index = node_index
		self.node_chain_id = -1
		self.next = None
		self.parents = []
		self.visited = False


class Tree:
	def __init__(self):
		self.head_pointer = []
		self.tail_pointers = []
		self.node_levels = []
		self.node_level_index_pair = []


def construct_tree(sorted_idx, all_nodes, neighbors):
	tree = Tree()
	pixel_number = len(sorted_idx)
	tree.node_levels = [-1] * pixel_number
	na = [False] * pixel_number
	ids = list(range(pixel_number))
	# traverse
	for cur_idx in tqdm(sorted_idx, desc='Construct tree'):
		neighbor_chain_id = []
		neighbor_tails = []

		# traverse the neighborhood
		for neighbor_id in neighbors[cur_idx]:
			if na[neighbor_id] is False:
				if all_nodes[neighbor_id].node_chain_id != -1 and neighbor_id != cur_idx:
					tail_node = tree.tail_pointers[all_nodes[neighbor_id].node_chain_id]
					while tail_node.next is not None:
						if tail_node.next.node_chain_id < tail_node.node_chain_id:
							tail_node = tree.tail_pointers[tail_node.next.node_chain_id]
						else:
							tail_node = tail_node.next

					neighbor_is_new_list = True
					for n in range(len(neighbor_tails)):
						if neighbor_tails[n].node_index == tail_node.node_index:
							neighbor_is_new_list = False
							break
					if neighbor_is_new_list:
						neighbor_tails.append(tail_node)

					neighbor_is_same_chain_id = False
					for m in range(len(neighbor_chain_id)):
						if tail_node.node_chain_id == neighbor_chain_id[m]:
							neighbor_is_same_chain_id = True
							break
					if not neighbor_is_same_chain_id:
						neighbor_chain_id.append(tail_node.node_chain_id)

		node_level = 0
		if len(neighbor_tails) != 0:
			for m in range(len(neighbor_tails)):
				neighbor_tails[m].next = all_nodes[cur_idx]
				all_nodes[cur_idx].parents.append(neighbor_tails[m])
				if tree.node_levels[ids[neighbor_tails[m].node_index]] > node_level:
					node_level = tree.node_levels[ids[neighbor_tails[m].node_index]]
			tree.node_levels[cur_idx] = node_level + 1
			min_neighbor_chain_id = neighbor_chain_id[0]
			for n in range(1, len(neighbor_chain_id)):
				if min_neighbor_chain_id > neighbor_chain_id[n]:
					min_neighbor_chain_id = neighbor_chain_id[n]
			all_nodes[cur_idx].node_chain_id = min_neighbor_chain_id
			tree.tail_pointers[min_neighbor_chain_id] = all_nodes[cur_idx]
		else:
			tree.head_pointer.append(all_nodes[cur_idx])
			tree.tail_pointers.append(all_nodes[cur_idx])
			all_nodes[cur_idx].node_chain_id = len(tree.head_pointer) - 1
			tree.node_levels[cur_idx] = 0

	tree.node_level_index_pair = []
	for i in range(pixel_number):
		tree.node_level_index_pair.append((tree.node_levels[i], ids[i]))

	tree.node_level_index_pair = sorted(tree.node_level_index_pair)

	child2parent = []
	for i in range(len(all_nodes)):
		temp = [i]
		parent_number = len(all_nodes[i].parents)

		for j in range(parent_number):
			temp.append(all_nodes[i].parents[j].node_index)
		child2parent.append(temp)

	return child2parent


def get_tree(height, width, eta, i, dataset_id, uncertain_label_path='', uncertain_threshold=0.65):
	N = eta ** i

	elev_path = 'data/dataset' + str(dataset_id) + '/elev.tif'
	elev = Image.open(elev_path)
	elev = np.array(elev)

	pooling = nn.AvgPool2d(N, N)
	elev = pooling(torch.tensor(elev).unsqueeze(0)).squeeze(0).numpy()
	elev_f = elev.flatten()

	row_0 = height // N
	column_0 = width // N
	pixel_number = row_0 * column_0
	ids = range(pixel_number)

	# create neighbor list
	neighbors = []
	for cur_idx in ids:
		row = cur_idx // column_0
		column = cur_idx % column_0
		temp = []
		for j in range(max(0, row - 1), min(row_0 - 1, row + 1) + 1):
			for k in range(max(0, column - 1), min(column_0 - 1, column + 1) + 1):
				temp.append(j * column_0 + k)
		neighbors.append(temp)

	if len(uncertain_label_path) != 0:
		target2pixels, area2elevation, pixel2target = uncertain_mapping(column_0, eta,
																		uncertain_label_path,
																		uncertain_threshold,
																		elev,
																		'Processing tree node')
		target_ids = list(target2pixels.keys())
		new_ids = range(len(target2pixels))
		id_map = dict(zip(target_ids, new_ids))

		new_neighbors = []
		for key, value in tqdm(target2pixels.items(), desc='Get neighbor'):
			temp = []
			if len(value) == 1:
				assert key == value[0], "inconsistent mapping for uncertain area"
				temp += [id_map[pixel2target[n]] for n in neighbors[key]]
			else:
				for v in value:
					temp += [id_map[pixel2target[n]] for n in neighbors[v]]
			new_neighbors.append(list(set(temp)))

		# sort the index based on the elevation
		elevations = list(area2elevation.values())
		sorted_idx = [t[1] for t in sorted(list(zip(elevations, new_ids)))]

		# create a node for each area
		all_nodes = [Node(elevations[i], i) for i in new_ids]
		parents = construct_tree(sorted_idx, all_nodes, new_neighbors)

		return parents, new_neighbors
	else:
		# sort the index based on the elevation
		sorted_idx = [t[1] for t in sorted(list(zip(elev_f, ids)))]

		# create a node for each area
		all_nodes = [Node(elev_f[i], i) for i in ids]

		parents = construct_tree(sorted_idx, all_nodes, neighbors)
		return parents, neighbors
