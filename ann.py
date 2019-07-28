from random import shuffle, seed
from math import fabs, pi

from os import listdir

import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class Net(nn.Module):
	def __init__(self, channels=10):
		super(Net, self).__init__()

		self.inps = 3

		self.lay1 = channels
		self.lay2 = channels
		self.lay3 = channels
		self.outs = 1

		self.fc1 = nn.Linear(self.inps, self.lay1)
		self.d1 = nn.Dropout(p=0.0)
		self.fc2 = nn.Linear(self.lay1, self.lay2)
		self.d2 = nn.Dropout(p=0.0)
		#self.fc3 = nn.Linear(self.lay2, self.lay3)
		#self.d3 = nn.Dropout(p=0.04)
		self.fc4 = nn.Linear(self.lay3, self.outs)

	def forward(self, x):
		x = self.d1(F.relu(self.fc1(x)))
		x = self.d2(F.relu(self.fc2(x)))
		#x = F.relu(self.fc3(x))
		x = self.fc4(x)

		return x


def get_itd(left, right, sample_rate=1.0/44100.0, p=0.9):
	left_mag = [fabs(el) for el in left]
	right_mag = [fabs(el) for el in right]
	l_max = max(left_mag)
	r_max = max(right_mag)

	right_onset = None
	left_onset = None

	assert len(left) == len(right)
	for i in range(len(left)):
		if left_mag[i] >= 0.9*l_max and left_onset is None:
			left_onset = i*sample_rate
		if right_mag[i] >= 0.9*r_max and right_onset is None:
			right_onset = i*sample_rate

		if left_onset is not None and right_onset is not None:
			break

	return left_onset-right_onset


def get_data(ids, head_sizes):
	print("Reading data")
	itds = []
	inps = []
	i = 0
	for id in ids:
		print(id)
		head_size = head_sizes[i]
		files = listdir("itd_data/"+id)
		for file in files:
			with open("itd_data/"+id+"/"+file) as f:
				left = []
				right = []
				f.readline()
				line = f.readline().split(" ")
				azi = float(line[1])
				line = f.readline().split(" ")
				ele = float(line[1])

				f.readline()
				f.readline()

				while True:
					line = f.readline().split("\t")
					if len(line) == 2:
						left.append(float(line[0]))
						right.append(float(line[1]))
					else:
						break
			itds.append([44100*get_itd(left, right)])
			inps.append([azi*pi/180.0, ele*pi/180.0, head_size])
		i += 1

	assert len(itds) == len(inps)
	n = len(inps)
	idxs = [i for i in range(n)]
	shuffle(idxs)

	itds = [itds[idx] for idx in idxs]
	inps = [inps[idx] for idx in idxs]

	return inps, itds


if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)
	anthro_mat = sio.loadmat('itd_data/anthropometry/anthro.mat')

	# Get the IDs. ach should have 3 numbers.
	head_sizes = [x[0] for x in anthro_mat['X']]
	should_keep = [size > 0 for size in head_sizes]

	seed(0)
	IDs = ['003', '008', '009', '010', '011', '012', '015', '017', '018', '019', '020', '021', '027', '028', '033', '040', '044', '048', '050', '051', '058', '059', '060', '061', '065', '119', '124', '126', '127', '131', '133', '134', '135', '137', '147', '148', '152', '153', '154', '155', '156', '158', '162', '163', '165']

	IDs = list(filter(lambda x: should_keep[IDs.index(x)], IDs))
	head_sizes = list(filter(lambda x: x > 0, head_sizes))

	idxs = [i for i in range(len(IDs))]
	shuffle(idxs)
	print(len(IDs))
	assert len(IDs) == 45 - 8
	assert len(head_sizes) == len(IDs)
	train_IDS = [IDs[idxs[i]] for i in range(27)]
	test_IDS = [IDs[idxs[i]] for i in range(27, len(IDs))]

	train_head_sizes = [head_sizes[idxs[i]] for i in range(27)]
	test_head_sizes = [head_sizes[idxs[i]] for i in range(27, len(IDs))]

	inps, itds = get_data(train_IDS, train_head_sizes)

	errors = []
	# for channels in range(5, 8):
	for channels in range(10, 11):
		net = Net(channels)
		net.train(True)

		crit = nn.MSELoss()
		optimizer = optim.Adam(net.parameters(), lr=1.0e-4, weight_decay=0.1)
		print("Starting training")

		c = 0
		batchsize = 16
		inps_tensor = [None]*batchsize
		outs_true = [None]*batchsize
		for epoch in range(1000):
			loss_epoch = 0.0

			assert len(inps) == len(itds)
			idx = [i for i in range(len(inps))]
			shuffle(idx)
			for i in range(len(inps)):
				inps_tensor[c] = inps[idx[i]]
				outs_true[c] = itds[idx[i]]

				# inps_tensor.append(inps[i])
				# outs_true.append(itds[i])

				c += 1
				if c == batchsize:
					optimizer.zero_grad()

					inps_tensor = torch.DoubleTensor(inps_tensor)
					outs_true = torch.DoubleTensor(outs_true)
					outs = net(inps_tensor)

					loss = crit(outs, outs_true)
					loss.backward()
					loss_epoch += loss.item()
					optimizer.step()

					inps_tensor = [None]*batchsize
					outs_true = [None]*batchsize
					c = 0
			print(channels, epoch, loss_epoch)

		# Validation
		print("Starting Validation (on train set)")
		net.train(False)
		error = 0.0
		for i in range(len(inps)):
			inps_tensor = torch.DoubleTensor(inps[i])
			outs_true = torch.DoubleTensor(itds[i])
			outs = net(inps_tensor)

			error += fabs(outs.item()-outs_true)
		errors.append(error/len(inps))
		print("Error of", channels, "channels:", error/len(inps))

		print("Starting Validation (on test set)")
		net.train(False)
		error = 0.0
		inps, itds = get_data(test_IDS, test_head_sizes)

		for i in range(len(inps)):
			inps_tensor = torch.DoubleTensor(inps[i])
			outs_true = torch.DoubleTensor(itds[i])
			outs = net(inps_tensor)

			#print(outs[0], outs_true[0])

			error += fabs(outs.item()-outs_true)
		# errors.append(error/len(inps))
		print("Error of", channels, "channels:", error/len(inps))

		#torch.save(net, "ANN")
	print(errors)

	# import matplotlib.pyplot as plt
	# plt.plot(errors)
	# plt.show()

	import pickle
	pickle.dump(net, open('net_weight01_nr2.p', 'wb'))

	torch.save(net, 'net_weight01_nr2')

	"""Error of 10 channels: 2.0205890134210533
	Starting Validation (on test set)
	Reading data
	163
	135
	152
	051
	154
	155
	061
	018
	137
	134
	Error of 10 channels: 2.3101025420778467
	[2.0205890134210533]"""
