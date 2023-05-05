import os.path as osp

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.utils import subgraph, k_hop_subgraph
import random
import numpy as np
import scipy.sparse as sp
import copy
# from graph_augs import Graph_Augmentor
"""
	Functions to help load the graph data
"""

def read_file(folder, name, dtype=None):
	path = osp.join(folder, '{}.txt'.format(name))
	return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
	"""
	PyG util code to create graph batches
	"""

	node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
	node_slice = torch.cat([torch.tensor([0]), node_slice])

	row, _ = data.edge_index
	edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
	edge_slice = torch.cat([torch.tensor([0]), edge_slice])

	# Edge indices should start at zero for every graph.
	data.edge_index -= node_slice[batch[row]].unsqueeze(0)
	data.__num_nodes__ = torch.bincount(batch).tolist()

	slices = {'edge_index': edge_slice}
	if data.x is not None:
		slices['x'] = node_slice
	if data.edge_attr is not None:
		slices['edge_attr'] = edge_slice
	if data.y is not None:
		if data.y.size(0) == batch.size(0):
			slices['y'] = node_slice
		else:
			slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

	return data, slices


def read_graph_data(folder, feature):
	"""
	PyG util code to create PyG data instance from raw graph data
	"""
	print(feature)
	node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
	# node_attributes1 = sp.load_npz(folder + f'new_{feature}_feature.npz')
	edge_index = read_file(folder, 'A', torch.long).t()
	node_graph_id = np.load(folder + 'node_graph_id.npy')
	graph_labels = np.load(folder + 'graph_labels.npy')


	edge_attr = None
	x = torch.from_numpy(node_attributes.todense()).to(torch.float)
	node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
	y = torch.from_numpy(graph_labels).to(torch.long)
	_, y = y.unique(sorted=True, return_inverse=True)

	num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
	edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
	edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
	data, slices = split(data, node_graph_id)
	return data, slices


class ToUndirected:
	def __init__(self):
		"""
		PyG util code to transform the graph to the undirected graph
		"""
		pass

	def __call__(self, data):
		edge_attr = None
		edge_index = to_undirected(data.edge_index, data.x.size(0))
		num_nodes = edge_index.max().item() + 1 if data.x is None else data.x.size(0)
		# edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
		edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
		data.edge_index = edge_index
		data.edge_attr = edge_attr
		return data


class DropEdge:
	def __init__(self, tddroprate):
		"""
		Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
		1) Generate TD and BU edge indices
		2) Drop out edges
		Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
		"""
		self.tddroprate = tddroprate

	def __call__(self, data):
		#
		# if self.tddroprate > 0:
		# 	row = list(edge_index[0])
		# 	col = list(edge_index[1])
		# 	length = len(row)
		# 	poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
		# 	poslist = sorted(poslist)
		# 	row = list(np.array(row)[poslist])
		# 	col = list(np.array(col)[poslist])
		# 	new_edgeindex = [row, col]
		# else:
		# 	new_edgeindex = edge_index
		#
		# burow = list(edge_index[1])
		# bucol = list(edge_index[0])
		# if self.budroprate > 0:
		# 	length = len(burow)
		# 	poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
		# 	poslist = sorted(poslist)
		# 	row = list(np.array(burow)[poslist])
		# 	col = list(np.array(bucol)[poslist])
		# 	bunew_edgeindex = [row, col]
		# else:
		# 	bunew_edgeindex = [burow, bucol]
		#
		# # data.x = x
		# data.edge_index = torch.LongTensor(new_edgeindex)
		# data.BU_edge_index = torch.LongTensor(bunew_edgeindex)
		data.root = torch.FloatTensor(data.x[0])
		data.root_index = torch.LongTensor([0])

		# 边
		node_num, _ = data.x.size()
		_, edge_num = data.edge_index.size()
		permute_num = int(edge_num * self.tddroprate)
		if permute_num == 0:
			permute_num = 1
		edge_index = data.edge_index
		edge_index = to_undirected(edge_index, data.x.size(0))
		unif = torch.ones(2, node_num)
		add_edge_idx = unif.multinomial(permute_num, replacement=True).to(data.x.device)
		# # 随机抽样
		unif = torch.ones(edge_num)
		keep_edge_idx = unif.multinomial((edge_num - permute_num), replacement=False)
		edge_index = torch.cat((edge_index[:, keep_edge_idx], add_edge_idx), dim=1)
		data.edge_index = edge_index

		# 点
		# if self.tddroprate > 0:
		# 	mask_num = int(data.num_nodes * self.tddroprate)
		# 	if mask_num == 0:
		# 		mask_num = 1
		# 	unif = torch.ones(data.num_nodes)
		# 	unif[data.root_index] = 0
		# 	mask_idx = unif.multinomial(mask_num, replacement=False)
		# 	shape = data.x.mean(dim=0).shape
		# 	token = torch.rand(shape[0])
		# 	#data.x.mean(dim=0)
		# 	data.x[mask_idx] = token
		return data

class AddEdge:
	def __init__(self, tddroprate):
		"""
		Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
		1) Generate TD and BU edge indices
		2) Drop out edges
		Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
		"""
		self.tddroprate = tddroprate

	def __call__(self, data):
		data.root = torch.FloatTensor(data.x[0])
		data.root_index = torch.LongTensor([0])
		# 边
		node_num, _ = data.x.size()
		_, edge_num = data.edge_index.size()
		permute_num = int(edge_num * self.tddroprate)
		if permute_num == 0:
			permute_num = 1
		edge_index = data.edge_index
		edge_index = to_undirected(edge_index, data.x.size(0))
		unif = torch.ones(2, node_num)
		add_edge_idx = unif.multinomial(permute_num, replacement=True).to(data.x.device)
		# # 随机抽样
		unif = torch.ones(edge_num)
		edge_index = torch.cat((edge_index, add_edge_idx), dim=1)
		data.edge_index = edge_index

		# 点
		# if self.tddroprate > 0:
		# 	mask_num = int(data.num_nodes * self.tddroprate)
		# 	if mask_num == 0:
		# 		mask_num = 1
		# 	unif = torch.ones(data.num_nodes)
		# 	unif[data.root_index] = 0
		# 	mask_idx = unif.multinomial(mask_num, replacement=False)
		# 	shape = data.x.mean(dim=0).shape
		# 	token = torch.rand(shape[0])
		# 	#data.x.mean(dim=0)
		# 	data.x[mask_idx] = token
		return data

class NodeDrop(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, data, aug_ratio):
		# data = copy.deepcopy(data)
		x, edge_index = data.x, data.edge_index

		num_nodes = x.size()[0]
		drop_num = int(num_nodes * aug_ratio)
		keep_num = num_nodes - drop_num

		# 随机选
		# torch.multinomial(weights, 4, replacement=False)
		keep_idx = torch.randperm(num_nodes)[:keep_num]
		# edge_index, _ = subgraph(keep_idx, edge_index)
		drop_idx = torch.ones(x.shape[0], dtype=bool)
		drop_idx[keep_idx] = False

		# 按照权重来选
		# root 不能被dropout
		drop_idx[0] = False

		x[drop_idx] = 0  # x 对应的特征变成0就 = dropout
		return x, edge_index

class Adaptive:
	def __init__(self):
		"""
		Data 这里的每个data都是一个小图
		"""
		self.Nodedrop = NodeDrop()

	# def NodeDrop(self, data, aug_ratio):
	# 	data = copy.deepcopy(data)
	# 	x, edge_index = data.x, data.edge_index
	# 	num_nodes = x.size()[0]
	# 	drop_num = int(num_nodes * aug_ratio)
	# 	keep_num = num_nodes - drop_num
	#
	# 	# 随机选
	# 	# torch.multinomial(weights, 4, replacement=False)
	# 	keep_idx = torch.randperm(num_nodes)[:keep_num]
	# 	# edge_index, _ = subgraph(keep_idx, edge_index)
	# 	drop_idx = torch.ones(x.shape[0], dtype=bool)
	# 	drop_idx[keep_idx] = False
	#
	# 	# 按照权重来选
	# 	# root 不能被dropout
	# 	drop_idx[0] = False
	#
	# 	x[drop_idx] = 0  # x 对应的特征变成0就 = dropout
	# 	return x, edge_index

	def EdgePerturb(self, data, aug_ratio):
		# data = copy.deepcopy(data)
		edge_index = data.edge_index
		if aug_ratio > 0:
			row = list(edge_index[0])
			col = list(edge_index[1])
			length = len(row)
			poslist = random.sample(range(length), int(length * (1 - aug_ratio)))
			poslist = sorted(poslist)
			row = list(np.array(row)[poslist])
			col = list(np.array(col)[poslist])
			new_edgeindex = [row, col]
		else:
			new_edgeindex = edge_index

		return data.x, torch.LongTensor(new_edgeindex)

	def AttrMask(self, data, aug_ratio):
		mask_num = int(data.num_nodes * aug_ratio)
		unif = torch.ones(data.num_nodes)
		mask_idx = unif.multinomial(mask_num, replacement=True)
		token = data.x.mean(dim=0)
		data.x[mask_idx] = token
		return data.x, data.edge_index

	def __call__(self, data):
		# x, edge_index = copy.copy(data.x), copy.copy(data.edge_index)
		x, edge_index = self.Nodedrop(data, 0.9)
		data.root = torch.FloatTensor(data.x[0])  # root 节点的向量表示
		data.root_index = torch.LongTensor([0])   # root 节点的index索引
		data.x = x
		data.edge_index = edge_index
		return data

class FNNDataset(InMemoryDataset):
	r"""
		The Graph datasets built upon FakeNewsNet data

	Args:
		root (string): Root directory where the dataset should be saved.
		name (string): The `name
			<https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
			dataset.
		transform (callable, optional): A function/transform that takes in an
			:obj:`torch_geometric.data.Data` object and returns a transformed
			version. The data object will be transformed before every access.
			(default: :obj:`None`)
		pre_transform (callable, optional): A function/transform that takes in
			an :obj:`torch_geometric.data.Data` object and returns a
			transformed version. The data object will be transformed before
			being saved to disk. (default: :obj:`None`)
		pre_filter (callable, optional): A function that takes in an
			:obj:`torch_geometric.data.Data` object and returns a boolean
			value, indicating whether the data object should be included in the
			final dataset. (default: :obj:`None`)
	"""

	def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
		self.name = name
		self.root = root
		self.feature = feature
		super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
		if not empty:
			self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0])

	@property
	def raw_dir(self):
		name = 'raw/'
		return osp.join(self.root, self.name, name)

	@property
	def processed_dir(self):
		name = 'processed/'
		return osp.join(self.root, self.name, name)

	@property
	def num_node_attributes(self):
		if self.data.x is None:
			return 0
		return self.data.x.size(1)

	@property
	def raw_file_names(self):
		names = ['node_graph_id', 'graph_labels']
		return ['{}.npy'.format(name) for name in names]

	@property
	def processed_file_names(self):
		if self.pre_filter is None:
			return f'{self.name[:3]}_data_{self.feature}.pt'
		else:
			return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

	def download(self):
		raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

	def process(self):

		self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

		if self.pre_filter is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [data for data in data_list if self.pre_filter(data)]
			self.data, self.slices = self.collate(data_list)

		if self.pre_transform is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [self.pre_transform(data) for data in data_list]
			self.data, self.slices = self.collate(data_list)

		# The fixed data split for benchmarking evaluation
		# train-val-test split is 20%-10%-70%
		self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
		self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)
		self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)

		torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])

	def __repr__(self):
		return '{}({})'.format(self.name, len(self))