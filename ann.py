from random import gauss, shuffle
from math import fabs

from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.inps = 3

		self.lay1 = 10
		self.lay2 = 10
		self.outs = 1

		self.fc1 = nn.Linear(self.inps, self.lay1)
		self.d1 = nn.Dropout(p=0.4)
		self.fc2 = nn.Linear(self.lay1, self.lay2)
		self.d2 = nn.Dropout(p=0.4)
		self.fc3 = nn.Linear(self.lay2, self.outs)

	def forward(self, x):
		x = self.d1(F.relu(self.fc1(x)))
		x = self.d2(F.relu(self.fc2(x)))
		x = self.fc3(x)

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
		if left_mag[i] >= 0.9*l_max and left_onset==None:
			left_onset = i*sample_rate
		if right_mag[i] >= 0.9*r_max and right_onset==None:
			right_onset = i*sample_rate

		if left_onset!=None and right_onset!=None:
			break

	return left_onset-right_onset

def get_head_size(id):
	return 1.0 # TODO: Write.

def get_data(ids):
	print("Reading data")
	itds = []
	inps = []
	for id in ids:
		head_size = get_head_size(id)
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
					if len(line)==2:
						left.append(float(line[0]))
						right.append(float(line[1]))
					else:
						break
			itds.append([get_itd(left, right)])
			inps.append([azi, ele, head_size])

	assert len(itds) == len(inps)
	n = len(inps)
	idxs = [i for i in range(n)]
	shuffle(idxs)

	itds = [itds[idx] for idx in idxs]
	inps = [inps[idx] for idx in idxs]

	return inps, itds


if __name__ == '__main__':
	inps, itds = get_data(['003'])

	from math import sin

	in_data = [torch.FloatTensor(inp) for inp in inps]
	out_data = [torch.FloatTensor(itd) for itd in itds]

	net = Net()

	crit = nn.MSELoss()
	optimizer = optim.Adam(net.parameters())
	print("Starting training")
	for epoch in range(1000):
		loss_sum = 0.0
		for i in range(len(in_data)):
			# TODO: Batchsize

			inps = in_data[i]
			outs_true = out_data[i]

			optimizer.zero_grad()

			outs = net(inps)

			loss = crit(outs, outs_true)
			loss_sum += loss.item()

			loss.backward()
			optimizer.step()

		if epoch%1==0:
			print(loss_sum)
